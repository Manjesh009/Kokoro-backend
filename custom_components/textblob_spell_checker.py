from typing import Dict, Text, Any, List
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from textblob import TextBlob


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=True
)
class CustomNLUComponent(GraphComponent):

    def __init__(self, config: Dict[Text, Any], resource: Resource) -> None:
        self.config = config
        self.resource = resource
        self.threshold = config.get("threshold", 0.5)
        self.batch_size = config.get("batch_size", 50)  # Default batch size

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, resource)

    def _process_batch(self, batch: List[Message]) -> None:
        """Processes a batch of training examples."""
        for example in batch:
            text = example.get("text")
            if text:
                try:
                    blob = TextBlob(text)
                    corrected_text = str(blob.correct())
                    example.set("text", corrected_text)
                except Exception as e:
                    print(f"Error processing text: {text[:50]}... - {e}")

    def train(self, training_data: TrainingData) -> Resource:
        examples = training_data.training_examples
        total_examples = len(examples)

        # Process in batches
        for i in range(0, total_examples, self.batch_size):
            batch = examples[i:i + self.batch_size]
            print(f"Processing batch {i // self.batch_size + 1} of {total_examples // self.batch_size + 1}...")
            self._process_batch(batch)

        return self.resource

    def process(self, messages: List[Message]) -> List[Message]:
        total_messages = len(messages)

        # Process in batches
        for i in range(0, total_messages, self.batch_size):
            batch = messages[i:i + self.batch_size]
            print(f"Processing batch {i // self.batch_size + 1} of {total_messages // self.batch_size + 1}...")
            self._process_batch(batch)

        return messages
