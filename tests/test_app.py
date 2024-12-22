import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

from unittest.mock import patch

from app import app


@pytest.fixture
def client():
    """Fixture pour initialiser un client de test Flask"""
    app.testing = True  # Met l'application en mode test
    with app.test_client() as client:
        yield client

def test_login_page(client):
    """Test de la route de connexion"""
    # Test GET de la page de connexion
    response = client.get('/login')
    assert response.status_code == 200

    # Test POST avec des données valides
    response = client.post('/login', data={
        'username': 'testuser',
        'password': 'testpassword'
    })
    assert response.status_code == 302  # Vérifie la redirection après une connexion réussie

    # Test POST avec des données invalides
    response = client.post('/login', data={
        'username': 'wronguser',  # Utilisateur incorrect
        'password': 'wrongpassword'
    })
    assert response.status_code == 302  # Redirection après une tentative échouée

def test_mlflow_connection():
    """Test simulé de la connexion à MLFlow"""
    with patch("requests.get") as mock_get:
        # Simule une réponse avec un statut HTTP 200
        mock_get.return_value.status_code = 200

        response = requests.get("http://127.0.0.1:5001")
        assert response.status_code == 200

        # Vérifie que la requête a bien été effectuée
        mock_get.assert_called_once_with("http://127.0.0.1:5001")
