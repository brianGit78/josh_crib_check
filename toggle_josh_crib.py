import requests
import asyncio
import httpx

class JoshAlertAsync:
    def __init__(self, home_assistant_url, ha_access_token, ha_entity_id, update_interval=120):
        """
        :param home_assistant_url: Base URL for Home Assistant's API, e.g. "http://192.168.1.10:8123/api/"
        :param ha_access_token: Long-lived access token for Home Assistant
        :param ha_entity_id: The entity ID you're toggling, e.g. "input_boolean.josh_in_crib"
        :param update_interval: How often (in seconds) to periodically refresh state from Home Assistant
        """
        # Configuration Variables
        self.home_assistant_url = home_assistant_url
        self.access_token = ha_access_token
        self.entity_id = ha_entity_id
        self.update_interval = update_interval

        # Internal State
        self._current_state = None    # e.g. "on" or "off"
        self._stop_event = asyncio.Event()

    async def start_periodic_check(self):
        """
        Call this to start the background task that periodically fetches the current state.
        """
        asyncio.create_task(self._periodic_state_updater())

    async def stop_periodic_check(self):
        """
        Call this to stop the background task gracefully.
        """
        self._stop_event.set()

    async def _periodic_state_updater(self):
        """
        Background task: periodically calls _refresh_state() to stay in sync with Home Assistant.
        """
        while not self._stop_event.is_set():
            await self._refresh_state()
            await asyncio.sleep(self.update_interval)

    async def _refresh_state(self):
        """
        Actually fetches the state from Home Assistant and updates self._current_state.
        """
        url = f"{self.home_assistant_url}states/{self.entity_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            json_resp = response.json()
            self._current_state = json_resp.get("state")
        except httpx.HTTPError as e:
            print(f"Error fetching state for {self.entity_id}: {e}")

    async def turn_on_helper(self):
        """
        Only calls Home Assistant 'turn_on' if _current_state != 'on'.
        """
        if self._current_state != "on":
            await self._call_service("turn_on")
            # Assume success; update local state
            self._current_state = "on"

    async def turn_off_helper(self):
        """
        Only calls Home Assistant 'turn_off' if _current_state != 'off'.
        """
        if self._current_state != "off":
            await self._call_service("turn_off")
            # Assume success; update local state
            self._current_state = "off"

    async def _call_service(self, action):
        """
        Generic method to POST to Home Assistant for turning on/off the entity.
        :param action: "turn_on" or "turn_off" (service name)
        """
        url = f"{self.home_assistant_url}services/input_boolean/{action}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        payload = {"entity_id": self.entity_id}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"Error calling service {action} on {self.entity_id}: {e}")


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
