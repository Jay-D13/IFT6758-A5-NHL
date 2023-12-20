import requests

class LiveGameClient:
    
    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1/gamecenter"
        self.event_tracker = set()
        self.last_event_id = None  # Stocke l'ID du dernier événement traité
        self.home_team = None
        self.away_team = None
        self.current_period = None
        self.time_remaining = None
        self.home_score = 0
        self.away_score = 0
        self.total_xg_home = 0
        self.total_xg_away = 0

    def get_live_game_events(self, game_id):
        url = f"{self.base_url}/{game_id}/play-by-play"
        params = {'lastEventId': self.last_event_id} if self.last_event_id else {}

        rappelle = requests.get(url, params=params)
        
        if rappelle.status_code == 200:
            donnee_live_match = rappelle.json()
            events = donnee_live_match.get("data", {}).get("gameData", {}).get("game", {}).get("plays", [])
            new_events = [event for event in events if event['eventID'] not in self.event_tracker]
            self.event_tracker.update(event['eventID'] for event in new_events)
        
        if new_events:
            self.last_event_id = new_events[-1]['eventID']
            return new_events
        else:
            # Gérez les erreurs de requête ici
            rappelle.raise_for_status()

    def process_live_events(self, live_events):
        # Traitez les nouveaux événements
        for event in live_events:
            # Produisez les caractéristiques requises par le modèle
            features = self.extract_features(event)

            # Mise à jour des statistiques du match
            self.update_game_stats(features)

            # Faites une demande au service de prédiction
            prediction = self.make_prediction(features)

            # Affichez les informations requises
            self.display_game_info()

    def extract_features(self, live_event):
        # Extraire les caractéristiques des événements
        if live_event['result']['eventTypeId'] in ['GOAL', 'SHOT']:
            features = {
                'event_type': 'SHOT_GOAL',  # Utilisez un nom commun pour les deux types
                'team': live_event['team']['triCode'],
                'period': live_event['about']['period'],
                'time_remaining': live_event['about']['periodTimeRemaining'],
                'shooter': live_event['players'][0]['player']['fullName'],
                'result': live_event['result']['eventTypeId'],
            }
            return features
        else:
            # Pour d'autres types d'événements, retournez un dictionnaire vide
            return {}

    def make_prediction(self, features):
        # Faites une demande au service Flask pour obtenir la prédiction
        url = "http://127.0.0.1:<PORT>/predict"
        data = {'features': [features]}  # Assurez-vous que les données sont correctement formatées
        response = requests.post(url, json=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction'][0]
            return prediction
        else:
            # Gérez les erreurs de requête ici
            response.raise_for_status()

    def update_game_stats(self, features):
        # Mise à jour des statistiques du match
        if features.get('event_type') == 'SHOT_GOAL':
            team = features.get('team')
            result = features.get('result')

            if result == 'GOAL':
                if team == self.home_team:
                    self.home_score += 1
                elif team == self.away_team:
                    self.away_score += 1

            # Mise à jour des buts attendus (xG)
            xg = self.make_prediction(features)
            if team == self.home_team:
                self.total_xg_home += xg
            elif team == self.away_team:
                self.total_xg_away += xg

            # Mise à jour de la période et du temps restant
            self.current_period = features.get('period')
            self.time_remaining = features.get('time_remaining')

    def display_game_info(self):
        # Affichez les informations requises
        print("Équipes:", self.home_team, "vs", self.away_team)
        print("Période:", self.current_period)
        print("Temps restant dans la période:", self.time_remaining)
        print("Score actuel:", f"{self.home_team} {self.home_score} - {self.away_team} {self.away_score}")
        print("Somme des buts attendus (xG) -", f"{self.home_team}: {self.total_xg_home}, {self.away_team}: {self.total_xg_away}")
        print("Différence entre le score actuel et la somme des buts attendus:", f"{self.home_team}: {self.home_score - self.total_xg_home}, {self.away_team}: {self.away_score - self.total_xg_away}")