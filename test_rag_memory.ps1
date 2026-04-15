# test_rag_memory.ps1
$session = "memtest-rag"

.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --question "What does P0300 usually indicate?" `
  --show-route `
  --show-details

.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --question "What should I check next based on that?" `
  --show-route `
  --show-details