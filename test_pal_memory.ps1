# test_pal_memory.ps1
$session = "memtest-pal"
$csv = "data/obdiidata/drive1.csv"

.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What is the average RPM?" `
  --show-route `
  --show-details

.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "How does that compare to idle?" `
  --show-route `
  --show-details