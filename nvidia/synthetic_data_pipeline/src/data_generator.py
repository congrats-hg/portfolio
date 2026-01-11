"""
Synthetic Data Generator
========================

Generates synthetic instruction-response pairs using LLMs.

Key techniques:
1. Self-Instruct: Generate new instructions from seed examples
2. Evol-Instruct: Evolve simple instructions into complex ones
3. Response generation with multiple sampling strategies

Reference:
- NVIDIA Nemotron-4 340B Instruct for synthetic generation
- Self-Instruct paper: "Self-Instruct: Aligning Language Model with Self Generated Instructions"
- Evol-Instruct: "WizardLM: Empowering Large Language Models to Follow Complex Instructions"
"""

import torch
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import random
import json
from pathlib import Path
from tqdm import tqdm
import re


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    num_instructions_per_seed: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512
    num_response_samples: int = 1
    evolution_depth: int = 2  # For Evol-Instruct
    batch_size: int = 8


@dataclass
class SyntheticSample:
    """A single synthetic data sample."""
    instruction: str
    response: str
    seed_instruction: Optional[str] = None
    evolution_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "seed_instruction": self.seed_instruction,
            "evolution_history": self.evolution_history,
            "metadata": self.metadata,
        }


class InstructionGenerator:
    """
    Generates diverse instructions from seed examples.

    Implements Self-Instruct methodology:
    1. Start with seed instructions
    2. Use LLM to generate similar but novel instructions
    3. Filter for quality and diversity
    """

    # Prompt templates for instruction generation
    GENERATION_PROMPT = """You are an AI assistant helping to create diverse instructions for training.

Here are some example instructions:
{seed_examples}

Generate {num_new} new and diverse instructions that are different from the examples above.
Each instruction should be clear, specific, and useful.

Requirements:
- Instructions should cover various topics and difficulty levels
- Avoid repetitive patterns
- Be creative but practical
- Each instruction on a new line, numbered

New Instructions:"""

    EVOLUTION_PROMPTS = {
        "add_constraints": """Make this instruction more complex by adding constraints or requirements:
Original: {instruction}
More complex version:""",

        "deepen": """Make this instruction require deeper thinking or more steps:
Original: {instruction}
Deeper version:""",

        "concretize": """Make this instruction more specific and concrete:
Original: {instruction}
More specific version:""",

        "broaden": """Generalize this instruction to be applicable to more situations:
Original: {instruction}
Broader version:""",

        "combine": """Combine this instruction with a related concept to create a multi-part task:
Original: {instruction}
Combined version:""",
    }

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GenerationConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate_from_seeds(
        self,
        seed_instructions: List[str],
        num_to_generate: int = 10,
    ) -> List[str]:
        """
        Generate new instructions from seed examples.

        Args:
            seed_instructions: List of seed instructions
            num_to_generate: Number of new instructions to generate

        Returns:
            List of generated instructions
        """
        # Format seed examples
        seed_examples = "\n".join([
            f"{i+1}. {inst}" for i, inst in enumerate(seed_instructions[:5])
        ])

        prompt = self.GENERATION_PROMPT.format(
            seed_examples=seed_examples,
            num_new=num_to_generate
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract instructions from generated text
        instructions = self._parse_instructions(generated_text)

        return instructions

    def _parse_instructions(self, text: str) -> List[str]:
        """Parse numbered instructions from generated text."""
        instructions = []

        # Look for numbered items
        pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            instruction = match.strip()
            if len(instruction) > 10:  # Filter out too short
                instructions.append(instruction)

        return instructions

    @torch.no_grad()
    def evolve_instruction(
        self,
        instruction: str,
        evolution_type: Optional[str] = None,
    ) -> str:
        """
        Evolve an instruction to be more complex.

        Args:
            instruction: Original instruction
            evolution_type: Type of evolution (random if None)

        Returns:
            Evolved instruction
        """
        if evolution_type is None:
            evolution_type = random.choice(list(self.EVOLUTION_PROMPTS.keys()))

        prompt = self.EVOLUTION_PROMPTS[evolution_type].format(
            instruction=instruction
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def evolve_multiple_rounds(
        self,
        instruction: str,
        num_rounds: int = 2,
    ) -> tuple[str, List[str]]:
        """
        Apply multiple evolution rounds to an instruction.

        Returns:
            Tuple of (final_instruction, evolution_history)
        """
        history = [instruction]
        current = instruction

        evolution_types = list(self.EVOLUTION_PROMPTS.keys())
        random.shuffle(evolution_types)

        for i in range(num_rounds):
            evolution_type = evolution_types[i % len(evolution_types)]
            evolved = self.evolve_instruction(current, evolution_type)
            history.append(evolved)
            current = evolved

        return current, history


class SyntheticDataGenerator:
    """
    Full synthetic data generation pipeline.

    Generates instruction-response pairs following NVIDIA Nemotron methodology:
    1. Generate diverse instructions from seeds
    2. Optionally evolve instructions for complexity
    3. Generate responses for each instruction
    4. Package as training samples

    Example usage:
        >>> generator = SyntheticDataGenerator(model, tokenizer)
        >>> samples = generator.generate(
        ...     seed_file="prompts/seeds.json",
        ...     num_samples=1000
        ... )
        >>> generator.save(samples, "data/synthetic.jsonl")
    """

    RESPONSE_PROMPT = """You are a helpful AI assistant. Please respond to the following instruction thoughtfully and comprehensively.

Instruction: {instruction}

Response:"""

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GenerationConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = device

        self.instruction_generator = InstructionGenerator(
            model, tokenizer, config, device
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate_response(self, instruction: str) -> str:
        """Generate a response for an instruction."""
        prompt = self.RESPONSE_PROMPT.format(instruction=instruction)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def generate(
        self,
        seed_instructions: List[str],
        num_samples: int = 100,
        evolve: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SyntheticSample]:
        """
        Generate synthetic samples from seed instructions.

        Args:
            seed_instructions: List of seed instructions
            num_samples: Target number of samples to generate
            evolve: Whether to apply Evol-Instruct
            progress_callback: Optional callback(current, total)

        Returns:
            List of SyntheticSample objects
        """
        samples = []

        # Phase 1: Generate new instructions
        print("Phase 1: Generating instructions...")
        instructions_per_batch = self.config.num_instructions_per_seed
        num_batches = (num_samples + instructions_per_batch - 1) // instructions_per_batch

        all_instructions = []
        for batch_idx in tqdm(range(num_batches), desc="Generating instructions"):
            # Sample seeds for this batch
            batch_seeds = random.sample(
                seed_instructions,
                min(5, len(seed_instructions))
            )

            new_instructions = self.instruction_generator.generate_from_seeds(
                batch_seeds,
                num_to_generate=instructions_per_batch
            )
            all_instructions.extend(new_instructions)

            if len(all_instructions) >= num_samples:
                break

        all_instructions = all_instructions[:num_samples]
        print(f"Generated {len(all_instructions)} instructions")

        # Phase 2: Optionally evolve instructions
        if evolve:
            print("Phase 2: Evolving instructions...")
            evolved_data = []
            for inst in tqdm(all_instructions, desc="Evolving"):
                try:
                    evolved, history = self.instruction_generator.evolve_multiple_rounds(
                        inst,
                        num_rounds=self.config.evolution_depth
                    )
                    evolved_data.append((evolved, inst, history))
                except Exception as e:
                    evolved_data.append((inst, inst, [inst]))

            all_instructions = [(e[0], e[1], e[2]) for e in evolved_data]
        else:
            all_instructions = [(inst, inst, [inst]) for inst in all_instructions]

        # Phase 3: Generate responses
        print("Phase 3: Generating responses...")
        for i, (instruction, seed, history) in enumerate(tqdm(all_instructions, desc="Generating responses")):
            try:
                response = self.generate_response(instruction)

                sample = SyntheticSample(
                    instruction=instruction,
                    response=response,
                    seed_instruction=seed,
                    evolution_history=history,
                    metadata={
                        "evolved": evolve,
                        "generation_idx": i,
                    }
                )
                samples.append(sample)

                if progress_callback:
                    progress_callback(i + 1, len(all_instructions))

            except Exception as e:
                print(f"Error generating response for instruction {i}: {e}")
                continue

        return samples

    def generate_from_file(
        self,
        seed_file: str,
        num_samples: int = 100,
        evolve: bool = True,
    ) -> List[SyntheticSample]:
        """Load seeds from file and generate samples."""
        path = Path(seed_file)

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    seeds = data
                else:
                    seeds = data.get("instructions", data.get("seeds", []))
        elif path.suffix == ".txt":
            with open(path) as f:
                seeds = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return self.generate(seeds, num_samples, evolve)

    def save(
        self,
        samples: List[SyntheticSample],
        output_path: str,
        format: str = "jsonl",
    ):
        """Save generated samples to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
        elif format == "json":
            with open(path, 'w') as f:
                json.dump([s.to_dict() for s in samples], f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved {len(samples)} samples to {path}")


class TemplateBasedGenerator:
    """
    Template-based generation for specific domains.

    Uses Jinja2 templates to generate structured instructions,
    similar to NVIDIA's Data Designer framework.
    """

    def __init__(
        self,
        model,
        tokenizer,
        template_dir: str = "prompts",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.template_dir = Path(template_dir)
        self.device = device

        try:
            from jinja2 import Environment, FileSystemLoader
            self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        except ImportError:
            self.env = None
            print("Jinja2 not installed. Template-based generation disabled.")

    def generate_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        num_samples: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Generate samples using a template.

        Args:
            template_name: Name of Jinja2 template file
            variables: Variables to fill in template
            num_samples: Number of samples to generate

        Returns:
            List of generated samples
        """
        if self.env is None:
            raise RuntimeError("Jinja2 required for template generation")

        template = self.env.get_template(template_name)
        samples = []

        for _ in range(num_samples):
            # Render template with variables
            prompt = template.render(**variables)

            # Generate response
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            samples.append({
                "prompt": prompt,
                "response": response.strip(),
                "template": template_name,
                "variables": variables,
            })

        return samples
