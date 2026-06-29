import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_weather_london_status_and_temp_max():
    r = client.get("/weather", params={"city": "London"})
    assert r.status_code == 200
    body = r.json()
    assert "today" in body
    assert "temp_max" in body["today"]


def test_predict_london_status_and_forecast_list():
    r = client.get("/predict", params={"city": "London"})
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 7


def test_compare_london_tokyo_status_and_both_cities():
    r = client.get("/compare", params={"city_a": "London", "city_b": "Tokyo"})
    assert r.status_code == 200
    body = r.json()
    assert body["city_a"] is not None
    assert body["city_b"] is not None
    assert "temp_max" in body["city_a"]
    assert "temp_max" in body["city_b"]


def test_autocomplete_returns_list():
    r = client.get("/autocomplete", params={"query": "Ban"})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) > 0
