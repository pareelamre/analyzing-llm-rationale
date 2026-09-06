# Local Datastore emulator

T04's durable-store checks use the Google Cloud Datastore emulator. Start it
from a PowerShell with Java 21 available:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
gcloud beta emulators datastore start --host-port=127.0.0.1:8765 --consistency=1.0 --no-store-on-disk
```

Run the process-contention suite in a separate PowerShell:

```powershell
$env:DATASTORE_EMULATOR_HOST = '127.0.0.1:8765'
$env:DATASTORE_DATASET = 'foresea-twin-test'
$env:GOOGLE_CLOUD_PROJECT = 'foresea-twin-test'
$env:PYTHONPATH = 'src'
py -m unittest tests.test_twin_store_integration
```

The suite proves that two processes cannot consume the same final account
allocation and that a confirmed manual command and an autonomous command share
the durable account capacity. It also checks an expired lease receives a new
fence while the stale worker cannot progress the command.
