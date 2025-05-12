# fleet-simulator
Battery Fleet Simulator App


1. Optional
    * Setup a virtual environment: `python -m venv .venv-fleet`
    * Activate new environment: `source .venv-fleet/bin/activate`

2. Install requirements: `pip install -r requirements.txt`

3. Run app from terminal or IDE e.g. `python fleet_simulator.py` - Dash app will run locally on http://127.0.0.1:8050/


Note:

* The battery assets are loaded stored separately as a `.json` file with the following format:

```
{
    "LZ_NORTH": {
        "capacity_mwh": 10,
        "charge_rate_mw": 1,
        "discharge_rate_mw": 2,
        "efficiency": 0.9
    },
    "LZ_SOUTH": {
        "capacity_mwh": 2,
        "charge_rate_mw": 0.3,
        "discharge_rate_mw": 1,
        "efficiency": 0.95
    },
    "RAYBN": {
        "capacity_mwh": 3,
        "charge_rate_mw": 1,
        "discharge_rate_mw": 1,
        "efficiency": 0.7
    },
    "GAMBIT": {
        "capacity_mwh": 100,
        "charge_rate_mw": 10,
        "discharge_rate_mw": 50,
        "efficiency": 0.99
    }
}
```

* Each asset has the following properties: capacity, max charge rate, max discharge rate, and efficiency

* Everything else can be control in the app itself e.g.
    * Fleet Max Charge Rate (MW) 
    * Fleet Max Discharge Rate (MW)
    * Number of Simulation Days