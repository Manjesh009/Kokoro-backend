from typing import Any, List, Text, Dict
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from textblob import TextBlob

class ActionCorrectSpelling(Action):
    def name(self) -> Text:
        return "action_correct_spelling"

    def run(self, dispatcher, tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get('text')
        if user_input:
            blob = TextBlob(user_input)
            corrected_text = blob.correct()
            corrected_text_str = str(corrected_text)
            dispatcher.utter_message(text=f"Corrected text: {corrected_text_str}")
            return [SlotSet("corrected_text", corrected_text_str)]
        return []
    
class ActionTrackChestPain(Action):
    def name(self) -> str:
        return "action_track_chest_pain"

    def run(self, dispatcher, tracker, domain):
        chest_pain_responded = tracker.get_slot("chest_pain_responded")
        if chest_pain_responded is False:
            dispatcher.utter_message(response="utter_ask_for_medical_help")
            return [SlotSet("chest_pain_responded", True)]  
        else:
            dispatcher.utter_message(response="utter_follow_up_action")
            return []  