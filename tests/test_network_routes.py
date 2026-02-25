"""Tests for Flask network API routes.

Creates a minimal Flask app with a real NetworkDB and a mock startup object,
bypassing the login_required decorator and heavy satorilib imports.
"""

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest.mock as mock

import pytest
from flask import Flask, jsonify, request, session

# Import NetworkDB directly (avoid __init__.py chain)
_spec = importlib.util.spec_from_file_location(
    'network_db',
    os.path.join(os.path.dirname(__file__),
                 '..', 'neuron-lite', 'satorineuron', 'network_db.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NetworkDB = _mod.NetworkDB


class MockStartup:
    """Minimal startup mock with a real NetworkDB."""

    def __init__(self, db_path):
        self.networkDB = NetworkDB(db_path)
        self._published = []
        self._discover_result = []  # set per test

    def publishObservation(self, stream_name, value):
        """Track publishes instead of actually broadcasting."""
        self._published.append({'stream_name': stream_name, 'value': value})

    def discoverRelaySync(self, relay_url):
        """Return pre-configured discovery results."""
        return self._discover_result


def create_test_app(startup):
    """Build a minimal Flask app with just the network API routes."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-secret'
    app.config['TESTING'] = True

    def get_startup():
        return startup

    # ── Subscriptions ─────────────────────────────────────

    @app.route('/api/network/subscriptions', methods=['GET'])
    def api_network_subscriptions():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        show_all = request.args.get('all')
        subs = s.networkDB.get_all() if show_all else s.networkDB.get_active()
        for sub in subs:
            sub['predicting'] = s.networkDB.is_predicting(
                sub['stream_name'], sub['provider_pubkey'])
        return jsonify({'subscriptions': subs})

    @app.route('/api/network/subscribe', methods=['POST'])
    def api_network_subscribe():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        relay_url = data.get('relay_url', '')
        row_id = s.networkDB.subscribe(data, relay_url)
        return jsonify({'id': row_id})

    @app.route('/api/network/unsubscribe', methods=['POST'])
    def api_network_unsubscribe():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        s.networkDB.unsubscribe(data['stream_name'], data['nostr_pubkey'])
        return jsonify({'success': True})

    @app.route('/api/network/observations', methods=['GET'])
    def api_network_observations():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        stream_name = request.args.get('stream_name')
        provider_pubkey = request.args.get('provider_pubkey')
        if not stream_name or not provider_pubkey:
            return jsonify({'error': 'Missing params'}), 400
        observations = s.networkDB.get_observations(
            stream_name, provider_pubkey)
        predictions = s.networkDB.get_predictions(
            stream_name, provider_pubkey)
        return jsonify({
            'observations': observations,
            'predictions': predictions,
        })

    # ── Publications ──────────────────────────────────────

    @app.route('/api/network/publications', methods=['GET'])
    def api_network_publications():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        show_all = request.args.get('all')
        pubs = (s.networkDB.get_all_publications() if show_all
                else s.networkDB.get_active_publications())
        return jsonify({'publications': pubs})

    @app.route('/api/network/predict', methods=['POST'])
    def api_network_predict():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        stream_name = data.get('stream_name')
        pubkey = data.get('nostr_pubkey')
        pred_stream = stream_name + '_pred'
        s.networkDB.add_publication(
            pred_stream,
            source_stream_name=stream_name,
            source_provider_pubkey=pubkey)
        return jsonify({'success': True, 'stream_name': pred_stream})

    @app.route('/api/network/stop-predict', methods=['POST'])
    def api_network_stop_predict():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        stream_name = data.get('stream_name')
        pred_stream = stream_name + '_pred'
        s.networkDB.remove_publication(pred_stream)
        return jsonify({'success': True})

    @app.route('/api/network/publish', methods=['POST'])
    def api_network_publish():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        if not data or 'stream_name' not in data or 'value' not in data:
            return jsonify({'error': 'Missing stream_name or value'}), 400
        stream_name = data['stream_name']
        value = data['value']
        pubs = s.networkDB.get_active_publications()
        pub = next(
            (p for p in pubs if p['stream_name'] == stream_name), None)
        if not pub:
            return jsonify(
                {'error': f'No active publication: {stream_name}'}), 404
        s.publishObservation(stream_name, value)
        return jsonify({'success': True, 'stream_name': stream_name})

    # ── Data Sources ──────────────────────────────────────

    @app.route('/api/network/data-source', methods=['GET'])
    def api_network_data_source_get():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        stream_name = request.args.get('stream_name')
        if not stream_name:
            return jsonify({'error': 'Missing stream_name'}), 400
        ds = s.networkDB.get_data_source(stream_name)
        if not ds:
            return jsonify({'error': 'Not found'}), 404
        return jsonify({'data_source': ds})

    @app.route('/api/network/data-source', methods=['POST'])
    def api_network_data_source_create():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        if not data or not data.get('stream_name'):
            return jsonify({'error': 'Missing stream_name'}), 400
        url = data.get('url', '').strip()
        cadence = data.get('cadence_seconds') or 0
        parser_type = data.get('parser_type', '') if url else ''
        parser_config = data.get('parser_config', '') if url else ''
        if url and not parser_config:
            return jsonify(
                {'error': 'Parser config required when URL is set'}), 400
        s.networkDB.add_data_source(
            stream_name=data['stream_name'],
            url=url, cadence_seconds=cadence,
            parser_type=parser_type, parser_config=parser_config,
            name=data.get('name', ''),
            description=data.get('description', ''),
            method=data.get('method', 'GET'),
            headers=data.get('headers'))
        s.networkDB.add_publication(
            stream_name=data['stream_name'],
            name=data.get('name', ''),
            description=data.get('description', ''),
            cadence_seconds=cadence or None)
        return jsonify({'success': True})

    # ── Relays ────────────────────────────────────────────

    @app.route('/api/network/relays', methods=['GET'])
    def api_network_relays():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        relays = s.networkDB.get_relays()
        return jsonify({'relays': relays})

    @app.route('/api/network/relay', methods=['POST'])
    def api_network_relay_add():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        relay_url = data.get('relay_url', '').strip()
        if not relay_url:
            return jsonify({'error': 'Missing relay_url'}), 400
        s.networkDB.upsert_relay(relay_url)
        return jsonify({'success': True, 'relay_url': relay_url})

    @app.route('/api/network/relay', methods=['DELETE'])
    def api_network_relay_delete():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        data = request.get_json()
        relay_url = data.get('relay_url', '').strip()
        if not relay_url:
            return jsonify({'error': 'Missing relay_url'}), 400
        s.networkDB.delete_relay(relay_url)
        return jsonify({'success': True})

    # ── Data Source Test ──────────────────────────────────

    @app.route('/api/network/data-source/test', methods=['POST'])
    def api_network_data_source_test():
        import requests as http_requests
        import json as json_mod
        data = request.get_json()
        url = data.get('url', '').strip()
        method = data.get('method', 'GET').upper()
        headers = None
        if data.get('headers'):
            try:
                headers = json_mod.loads(data['headers'])
            except Exception:
                return jsonify({'error': 'Invalid headers JSON', 'raw': ''})
        parser_type = data.get('parser_type', 'json_path')
        parser_config = data.get('parser_config', '')
        try:
            if method == 'POST':
                resp = http_requests.post(url, headers=headers, timeout=15)
            else:
                resp = http_requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            raw = resp.text
        except Exception as e:
            return jsonify({'error': f'Fetch failed: {e}', 'raw': ''})
        try:
            if parser_type == 'json_path':
                obj = json_mod.loads(raw)
                for key in parser_config.split('.'):
                    if key.isdigit():
                        obj = obj[int(key)]
                    else:
                        obj = obj[key]
                value = str(obj)
            elif parser_type == 'python':
                local_vars = {'text': raw}
                exec_code = parser_config.strip()
                if 'return ' in exec_code and not exec_code.startswith('def '):
                    exec_code = ('def _parse(text):\n' +
                                 '\n'.join('    ' + l for l in exec_code.split('\n')) +
                                 '\n_result = _parse(text)')
                    exec(exec_code, {}, local_vars)
                    value = str(local_vars.get('_result', ''))
                else:
                    exec(exec_code, {}, local_vars)
                    value = str(local_vars.get('result', local_vars.get('_result', '')))
            else:
                value = ''
                return jsonify({'error': f'Unknown parser type: {parser_type}', 'raw': raw})
        except Exception as e:
            return jsonify({'error': f'Parse failed: {e}', 'raw': raw})
        return jsonify({'value': value, 'raw': raw})

    # ── Stream Discovery per Relay ────────────────────────

    @app.route('/api/network/streams/relay', methods=['GET'])
    def api_network_streams_relay():
        s = get_startup()
        if not s:
            return jsonify({'error': 'Startup not initialized'}), 503
        relay_url = request.args.get('url')
        if not relay_url:
            return jsonify({'error': 'Missing url parameter'}), 400
        streams = s.discoverRelaySync(relay_url)
        for st in streams:
            st['subscribed'] = s.networkDB.is_subscribed(
                st['stream_name'], st['nostr_pubkey'])
        return jsonify({'streams': streams})

    return app


@pytest.fixture
def app_and_startup():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_network.db')
        startup = MockStartup(db_path)
        app = create_test_app(startup)
        yield app, startup


@pytest.fixture
def client(app_and_startup):
    app, _ = app_and_startup
    return app.test_client()


@pytest.fixture
def startup(app_and_startup):
    _, s = app_and_startup
    return s


@pytest.fixture
def sample_stream():
    return {
        'stream_name': 'btc-price',
        'nostr_pubkey': 'abc123',
        'name': 'Bitcoin Price',
        'description': 'BTC/USD',
        'cadence_seconds': 3600,
        'price_per_obs': 0,
        'encrypted': False,
        'tags': ['bitcoin'],
        'relay_url': 'wss://relay1.example.com',
    }


# ── Subscriptions ─────────────────────────────────────────────────


class TestSubscriptionRoutes:

    def test_list_empty(self, client):
        resp = client.get('/api/network/subscriptions')
        assert resp.status_code == 200
        assert resp.json['subscriptions'] == []

    def test_subscribe_and_list(self, client, sample_stream):
        resp = client.post('/api/network/subscribe',
                           json=sample_stream)
        assert resp.status_code == 200
        assert resp.json['id'] > 0

        resp = client.get('/api/network/subscriptions')
        subs = resp.json['subscriptions']
        assert len(subs) == 1
        assert subs[0]['stream_name'] == 'btc-price'
        assert subs[0]['predicting'] is False

    def test_unsubscribe(self, client, sample_stream):
        client.post('/api/network/subscribe', json=sample_stream)
        resp = client.post('/api/network/unsubscribe',
                           json={'stream_name': 'btc-price',
                                 'nostr_pubkey': 'abc123'})
        assert resp.status_code == 200
        # Active list is empty
        resp = client.get('/api/network/subscriptions')
        assert len(resp.json['subscriptions']) == 0
        # All list has it
        resp = client.get('/api/network/subscriptions?all=1')
        assert len(resp.json['subscriptions']) == 1

    def test_predicting_flag(self, client, startup, sample_stream):
        client.post('/api/network/subscribe', json=sample_stream)
        startup.networkDB.add_publication(
            'btc-price_pred',
            source_stream_name='btc-price',
            source_provider_pubkey='abc123')
        resp = client.get('/api/network/subscriptions')
        assert resp.json['subscriptions'][0]['predicting'] is True


# ── Observations ──────────────────────────────────────────────────


class TestObservationRoutes:

    def test_missing_params(self, client):
        resp = client.get('/api/network/observations')
        assert resp.status_code == 400

    def test_empty_observations(self, client):
        resp = client.get(
            '/api/network/observations?stream_name=btc-price'
            '&provider_pubkey=abc123')
        assert resp.status_code == 200
        assert resp.json['observations'] == []
        assert resp.json['predictions'] == []

    def test_observations_with_data(self, client, startup):
        startup.networkDB.save_observation(
            'btc-price', 'abc123', '42000', 'evt1', seq_num=1)
        startup.networkDB.save_prediction(
            'btc-price', 'abc123', '42100', observation_seq=1)
        resp = client.get(
            '/api/network/observations?stream_name=btc-price'
            '&provider_pubkey=abc123')
        assert len(resp.json['observations']) == 1
        assert len(resp.json['predictions']) == 1


# ── Publications ──────────────────────────────────────────────────


class TestPublicationRoutes:

    def test_list_empty(self, client):
        resp = client.get('/api/network/publications')
        assert resp.status_code == 200
        assert resp.json['publications'] == []

    def test_predict_creates_publication(self, client, sample_stream):
        client.post('/api/network/subscribe', json=sample_stream)
        resp = client.post('/api/network/predict',
                           json={'stream_name': 'btc-price',
                                 'nostr_pubkey': 'abc123'})
        assert resp.status_code == 200
        assert resp.json['stream_name'] == 'btc-price_pred'

        resp = client.get('/api/network/publications')
        pubs = resp.json['publications']
        assert len(pubs) == 1
        assert pubs[0]['source_stream_name'] == 'btc-price'

    def test_stop_predict(self, client, sample_stream):
        client.post('/api/network/subscribe', json=sample_stream)
        client.post('/api/network/predict',
                    json={'stream_name': 'btc-price',
                          'nostr_pubkey': 'abc123'})
        resp = client.post('/api/network/stop-predict',
                           json={'stream_name': 'btc-price'})
        assert resp.status_code == 200
        resp = client.get('/api/network/publications')
        assert len(resp.json['publications']) == 0

    def test_publish_to_existing_publication(self, client, startup):
        startup.networkDB.add_publication('my-stream')
        resp = client.post('/api/network/publish',
                           json={'stream_name': 'my-stream',
                                 'value': '42.5'})
        assert resp.status_code == 200
        assert startup._published[-1] == {
            'stream_name': 'my-stream', 'value': '42.5'}

    def test_publish_missing_params(self, client):
        resp = client.post('/api/network/publish', json={'stream_name': 'x'})
        assert resp.status_code == 400

    def test_publish_no_publication(self, client):
        resp = client.post('/api/network/publish',
                           json={'stream_name': 'nonexistent',
                                 'value': '1'})
        assert resp.status_code == 404


# ── Data Sources ──────────────────────────────────────────────────


class TestDataSourceRoutes:

    def test_create_with_url(self, client):
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'btc-price',
            'url': 'https://api.example.com/price',
            'cadence_seconds': 3600,
            'parser_type': 'json_path',
            'parser_config': 'data.price',
        })
        assert resp.status_code == 200
        assert resp.json['success'] is True

    def test_create_also_makes_publication(self, client):
        client.post('/api/network/data-source', json={
            'stream_name': 'btc-price',
            'url': 'https://api.example.com',
            'cadence_seconds': 3600,
            'parser_type': 'json_path',
            'parser_config': 'price',
        })
        resp = client.get('/api/network/publications')
        pubs = resp.json['publications']
        assert len(pubs) == 1
        assert pubs[0]['stream_name'] == 'btc-price'

    def test_create_without_url(self, client):
        """External-push data source needs only stream_name."""
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'ext-stream',
        })
        assert resp.status_code == 200

    def test_create_url_without_parser_fails(self, client):
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'btc-price',
            'url': 'https://api.example.com',
            'cadence_seconds': 3600,
        })
        assert resp.status_code == 400
        assert 'Parser config required' in resp.json['error']

    def test_create_missing_stream_name(self, client):
        resp = client.post('/api/network/data-source', json={
            'url': 'https://example.com',
        })
        assert resp.status_code == 400

    def test_get_data_source(self, client):
        client.post('/api/network/data-source', json={
            'stream_name': 'btc-price',
            'url': 'https://api.example.com',
            'cadence_seconds': 3600,
            'parser_type': 'json_path',
            'parser_config': 'data.price',
            'name': 'BTC',
        })
        resp = client.get(
            '/api/network/data-source?stream_name=btc-price')
        assert resp.status_code == 200
        ds = resp.json['data_source']
        assert ds['stream_name'] == 'btc-price'
        assert ds['name'] == 'BTC'

    def test_get_data_source_not_found(self, client):
        resp = client.get(
            '/api/network/data-source?stream_name=nonexistent')
        assert resp.status_code == 404

    def test_get_data_source_missing_param(self, client):
        resp = client.get('/api/network/data-source')
        assert resp.status_code == 400

    def test_upsert_data_source(self, client):
        """Creating same stream_name twice updates instead of duplicating."""
        client.post('/api/network/data-source', json={
            'stream_name': 'btc-price',
            'url': 'https://old.com',
            'cadence_seconds': 60,
            'parser_type': 'json_path',
            'parser_config': 'old',
        })
        client.post('/api/network/data-source', json={
            'stream_name': 'btc-price',
            'url': 'https://new.com',
            'cadence_seconds': 120,
            'parser_type': 'json_path',
            'parser_config': 'new',
        })
        resp = client.get(
            '/api/network/data-source?stream_name=btc-price')
        assert resp.json['data_source']['url'] == 'https://new.com'


# ── Relays ────────────────────────────────────────────────────────


class TestRelayRoutes:

    def test_list_empty(self, client):
        resp = client.get('/api/network/relays')
        assert resp.status_code == 200
        assert resp.json['relays'] == []

    def test_add_relay(self, client):
        resp = client.post('/api/network/relay',
                           json={'relay_url': 'wss://relay1.example.com'})
        assert resp.status_code == 200
        resp = client.get('/api/network/relays')
        assert len(resp.json['relays']) == 1

    def test_add_relay_missing_url(self, client):
        resp = client.post('/api/network/relay', json={})
        assert resp.status_code == 400

    def test_delete_relay(self, client):
        client.post('/api/network/relay',
                    json={'relay_url': 'wss://relay1.example.com'})
        resp = client.delete('/api/network/relay',
                             json={'relay_url': 'wss://relay1.example.com'})
        assert resp.status_code == 200
        resp = client.get('/api/network/relays')
        assert len(resp.json['relays']) == 0


# ── End-to-end flows ─────────────────────────────────────────────


class TestEndToEnd:

    def test_subscribe_predict_observe(self, client, startup,
                                       sample_stream):
        """Full flow: subscribe → predict → observe → get data."""
        # Subscribe
        client.post('/api/network/subscribe', json=sample_stream)

        # Start predicting
        client.post('/api/network/predict',
                    json={'stream_name': 'btc-price',
                          'nostr_pubkey': 'abc123'})

        # Simulate observation + prediction
        startup.networkDB.save_observation(
            'btc-price', 'abc123', '42000', 'evt1', seq_num=1)
        startup.networkDB.save_prediction(
            'btc-price', 'abc123', '42100', observation_seq=1)

        # Verify data is accessible
        resp = client.get(
            '/api/network/observations?stream_name=btc-price'
            '&provider_pubkey=abc123')
        assert len(resp.json['observations']) == 1
        assert len(resp.json['predictions']) == 1

        # Verify prediction publication exists
        resp = client.get('/api/network/publications')
        pubs = resp.json['publications']
        assert any(p['stream_name'] == 'btc-price_pred' for p in pubs)

    def test_data_source_and_publish(self, client, startup):
        """Create external data source, then push data via publish."""
        # Create external data source (no URL)
        client.post('/api/network/data-source', json={
            'stream_name': 'my-sensor',
            'name': 'My Sensor',
        })

        # Publication should exist
        resp = client.get('/api/network/publications')
        assert any(
            p['stream_name'] == 'my-sensor'
            for p in resp.json['publications'])

        # Push data
        resp = client.post('/api/network/publish',
                           json={'stream_name': 'my-sensor',
                                 'value': '23.5'})
        assert resp.status_code == 200
        assert startup._published[-1]['value'] == '23.5'


# ── Data Source Test Endpoint ────────────────────────────────


class TestDataSourceTestRoute:

    def test_json_path_success(self, client):
        """Successful fetch + json_path parse returns value and raw."""
        mock_resp = mock.MagicMock()
        mock_resp.text = '{"data": {"price": 42.5}}'
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.get.return_value = mock_resp

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://api.example.com/price',
                'parser_type': 'json_path',
                'parser_config': 'data.price',
            })
        assert resp.status_code == 200
        assert resp.json['value'] == '42.5'
        assert 'raw' in resp.json

    def test_json_path_array_index(self, client):
        """json_path with numeric index traverses arrays."""
        mock_resp = mock.MagicMock()
        mock_resp.text = '{"markets": [{"price": 100}, {"price": 200}]}'
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.get.return_value = mock_resp

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://api.example.com',
                'parser_type': 'json_path',
                'parser_config': 'markets.1.price',
            })
        assert resp.status_code == 200
        assert resp.json['value'] == '200'

    def test_python_parser_with_return(self, client):
        """Python parser wraps return statements in a function."""
        mock_resp = mock.MagicMock()
        mock_resp.text = '{"val": 99}'
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.get.return_value = mock_resp

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://api.example.com',
                'parser_type': 'python',
                'parser_config': 'import json\nreturn str(json.loads(text)["val"])',
            })
        assert resp.status_code == 200
        assert resp.json['value'] == '99'

    def test_post_method(self, client):
        """method=POST uses requests.post."""
        mock_resp = mock.MagicMock()
        mock_resp.text = '{"x": 1}'
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.post.return_value = mock_resp

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://api.example.com',
                'method': 'POST',
                'parser_type': 'json_path',
                'parser_config': 'x',
            })
        assert resp.status_code == 200
        assert resp.json['value'] == '1'

    def test_fetch_failure(self, client):
        """Network error returns error with empty raw."""
        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.get.side_effect = Exception('Connection refused')

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://down.example.com',
                'parser_type': 'json_path',
                'parser_config': 'x',
            })
        assert resp.status_code == 200  # route returns 200 with error in body
        assert 'Fetch failed' in resp.json['error']
        assert resp.json['raw'] == ''

    def test_parse_failure(self, client):
        """Bad parser config returns error with raw body."""
        mock_resp = mock.MagicMock()
        mock_resp.text = '{"a": 1}'
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.get.return_value = mock_resp

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://api.example.com',
                'parser_type': 'json_path',
                'parser_config': 'nonexistent.key',
            })
        assert resp.status_code == 200
        assert 'Parse failed' in resp.json['error']
        assert resp.json['raw'] == '{"a": 1}'

    def test_invalid_headers_json(self, client):
        """Malformed headers JSON returns error immediately."""
        resp = client.post('/api/network/data-source/test', json={
            'url': 'https://api.example.com',
            'headers': 'not valid json{',
            'parser_type': 'json_path',
            'parser_config': 'x',
        })
        assert resp.status_code == 200
        assert 'Invalid headers JSON' in resp.json['error']

    def test_unknown_parser_type(self, client):
        """Unknown parser type returns error."""
        mock_resp = mock.MagicMock()
        mock_resp.text = 'hello'
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict(sys.modules, {'requests': mock.MagicMock()}):
            import requests as patched
            patched.get.return_value = mock_resp

            resp = client.post('/api/network/data-source/test', json={
                'url': 'https://api.example.com',
                'parser_type': 'xml',
                'parser_config': '/root',
            })
        assert resp.status_code == 200
        assert 'Unknown parser type' in resp.json['error']


# ── Stream Discovery per Relay ───────────────────────────────


class TestStreamDiscoveryRoute:

    def test_discover_streams(self, client, startup):
        """Returns discovered streams with subscribed flag."""
        startup._discover_result = [
            {'stream_name': 'btc-price', 'nostr_pubkey': 'pub1',
             'name': 'BTC', 'cadence_seconds': 3600},
            {'stream_name': 'eth-price', 'nostr_pubkey': 'pub2',
             'name': 'ETH', 'cadence_seconds': 1800},
        ]
        resp = client.get(
            '/api/network/streams/relay?url=wss://relay.example.com')
        assert resp.status_code == 200
        streams = resp.json['streams']
        assert len(streams) == 2
        assert streams[0]['subscribed'] is False
        assert streams[1]['subscribed'] is False

    def test_subscribed_flag_set(self, client, startup, sample_stream):
        """Subscribed streams get subscribed=True."""
        client.post('/api/network/subscribe', json=sample_stream)
        startup._discover_result = [
            {'stream_name': 'btc-price', 'nostr_pubkey': 'abc123',
             'name': 'BTC', 'cadence_seconds': 3600},
        ]
        resp = client.get(
            '/api/network/streams/relay?url=wss://relay.example.com')
        assert resp.json['streams'][0]['subscribed'] is True

    def test_missing_url_param(self, client):
        """Missing url parameter returns 400."""
        resp = client.get('/api/network/streams/relay')
        assert resp.status_code == 400
        assert 'Missing url' in resp.json['error']

    def test_empty_discovery(self, client, startup):
        """No streams found returns empty list."""
        startup._discover_result = []
        resp = client.get(
            '/api/network/streams/relay?url=wss://empty.example.com')
        assert resp.status_code == 200
        assert resp.json['streams'] == []


# ── CSV Bulk Upload (server-side validation) ─────────────────


class TestCsvBulkUpload:
    """Tests that mimic what the JS uploadCsv() function does:
    parse CSV rows and POST /api/network/data-source for each."""

    def test_basic_csv_row(self, client):
        """A well-formed CSV row creates data source + publication."""
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'csv-stream-1',
            'url': 'https://api.example.com/data',
            'cadence_seconds': 900,  # 15 min
            'parser_type': 'json_path',
            'parser_config': 'data.value',
        })
        assert resp.status_code == 200
        # Verify publication was created
        resp = client.get('/api/network/publications')
        names = [p['stream_name'] for p in resp.json['publications']]
        assert 'csv-stream-1' in names

    def test_multiple_csv_rows(self, client):
        """Multiple CSV rows create multiple data sources."""
        rows = [
            {'stream_name': 'csv-a', 'url': 'https://a.com',
             'cadence_seconds': 3600, 'parser_type': 'json_path',
             'parser_config': 'price'},
            {'stream_name': 'csv-b', 'url': 'https://b.com',
             'cadence_seconds': 1800, 'parser_type': 'json_path',
             'parser_config': 'value'},
            {'stream_name': 'csv-c'},  # external, no URL
        ]
        for row in rows:
            resp = client.post('/api/network/data-source', json=row)
            assert resp.status_code == 200

        resp = client.get('/api/network/publications')
        names = [p['stream_name'] for p in resp.json['publications']]
        assert 'csv-a' in names
        assert 'csv-b' in names
        assert 'csv-c' in names

    def test_missing_stream_name_rejected(self, client):
        """CSV row without stream_name is rejected."""
        resp = client.post('/api/network/data-source', json={
            'url': 'https://api.example.com',
            'parser_type': 'json_path',
            'parser_config': 'x',
        })
        assert resp.status_code == 400

    def test_url_without_parser_config_rejected(self, client):
        """CSV row with URL but no parser_config is rejected."""
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'csv-bad',
            'url': 'https://api.example.com',
            'cadence_seconds': 3600,
        })
        assert resp.status_code == 400
        assert 'Parser config required' in resp.json['error']

    def test_no_url_no_parser_ok(self, client):
        """External data source (no URL) doesn't need parser config."""
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'csv-ext',
        })
        assert resp.status_code == 200

    def test_cadence_from_csv_minutes(self, client):
        """JS converts cadence_minutes to cadence_seconds (×60)."""
        cadence_min = 15
        resp = client.post('/api/network/data-source', json={
            'stream_name': 'csv-cadence',
            'url': 'https://api.example.com',
            'cadence_seconds': cadence_min * 60,
            'parser_type': 'json_path',
            'parser_config': 'price',
        })
        assert resp.status_code == 200
        ds = client.get(
            '/api/network/data-source?stream_name=csv-cadence')
        assert ds.json['data_source']['cadence_seconds'] == 900
