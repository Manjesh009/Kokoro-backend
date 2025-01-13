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

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, resource)

    def train(self, training_data: TrainingData) -> Resource:
        for example in training_data.training_examples:
            text = example.get("text")
            if text:
                blob = TextBlob(text)
                corrected_text = str(blob.correct())
                example.set("text", corrected_text)
        return self.resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            text = example.get("text")
            if text:
                blob = TextBlob(text)
                corrected_text = str(blob.correct())
                example.set("text", corrected_text)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            text = message.get("text")
            if text:
                blob = TextBlob(text)
                corrected_text = str(blob.correct())
                message.set("text", corrected_text, add_to_output=True)
        return messages