import requests
#import creds

class JoshAlert:
    def __init__(self, home_assistant_url, ha_access_token, ha_entity_id):
        # Configuration Variables
        self.home_assistant_url = home_assistant_url
        self.access_token = ha_access_token
        self.entity_id = ha_entity_id

    def get_entity_state(self):
        url = f"{self.home_assistant_url}states/{self.entity_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers)
        return response.json().get("state")

    def turn_on_helper(self):
        if self.get_entity_state() != "on":
            url = f"{self.home_assistant_url}services/input_boolean/turn_on"
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
        if self.get_entity_state() != "off":
            url = f"{self.home_assistant_url}services/input_boolean/turn_off"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "entity_id": self.entity_id
            }
            response = requests.post(url, json=payload, headers=headers)
            #print("Turn Off Response:", response.text)
