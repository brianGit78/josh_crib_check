import requests
import creds

class JoshAlert:
    def __init__(self):
        # Configuration Variables
        self.home_assistant_url = creds.home_assistant_url
        self.access_token = creds.ha_access_token
        self.entity_id = creds.ha_entity_id

    def turn_on_helper(self):
        url = f"{self.home_assistant_url}turn_on"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "entity_id": self.entity_id
        }
        response = requests.post(url, json=payload, headers=headers)
        #print("Turn On Response:", response.text)

    def turn_off_helper(self):
        url = f"{self.home_assistant_url}turn_off"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "entity_id": self.entity_id
        }
        response = requests.post(url, json=payload, headers=headers)
        #print("Turn Off Response:", response.text)
