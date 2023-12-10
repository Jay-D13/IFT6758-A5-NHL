import requests

class LiveGameClient:
    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1/gamecenter"
        self.event_tracker = set()
        self.last_event_id = None  # Stocke l'ID du dernier événement traité

    def get_live_game_events(self, game_id):
        url = f"{self.base_url}/{game_id}/play-by-play"
        params = {'lastEventId': self.last_event_id} if self.last_event_id else {}

        rappelle = requests.get(url, params=params)
        
        if rappelle.status_code ==200:
            donnee_live_match = rappelle.json()
            events = donne_live_match.get("data": {}).get("gameData": {}).get("game", {}).get("plays", [])
            new_events = [event for event in events if event['eventID'] not in self.event_tracker]
            self.event_tracker.update(event['eventID'] for event in new_events)
        
        if new_events:
            self.last_event_id = new_events[-1]['eventID']

            return new_events
        else:
            # Gérez les erreurs de requête ici
            response.raise_for_status()

    def process_live_events(self, live_events):
        # Traitez les nouveaux événements selon vos besoins
        for event in live_events:
            # Produisez les caractéristiques requises par le modèle
            features = extract_features(event)

            # Faites une demande au service de prédiction
            prediction = make_prediction(features)