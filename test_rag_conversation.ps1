# test_rag_conversation.ps1
# Multi-turn RAG conversation tests — verifies that follow-up questions
# on fault codes carry context from the previous answer within the same session.

$docs = "knowledge/diagnostics/fault_codes_database.md"
$csv  = "data/obdiidata/drive1.csv"

# -----------------------------------------------------------------------
# Test 1: Fault code lookup then cause follow-up
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 1: Code lookup then cause follow-up ==========" -ForegroundColor Cyan
$session = "rag-conv-test-1"

Write-Host "`n[Turn 1] What does P0171 mean" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What does fault code P0171 mean?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] What causes it — should carry P0171 context from Turn 1" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What typically causes that fault?" `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 2: Description to code then follow-up explanation
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 2: Description to code then explanation ==========" -ForegroundColor Cyan
$session = "rag-conv-test-2"

Write-Host "`n[Turn 1] What code means oxygen sensor slow response" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "Which fault code relates to an oxygen sensor slow response?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] Full description of that code — context from Turn 1" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "Give me the full description of that code." `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 3: Hybrid — data observation then diagnostic follow-up
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 3: Hybrid data → diagnostic follow-up ==========" -ForegroundColor Cyan
$session = "rag-conv-test-3"

Write-Host "`n[Turn 1] PAL — short-term fuel trim average (data question)" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What is the average short-term fuel trim in drive1.csv?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] RAG — what fault code is associated with high fuel trim" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What fault codes are associated with high fuel trim readings?" `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 4: Multiple codes in sequence
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 4: Sequential code lookups ==========" -ForegroundColor Cyan
$session = "rag-conv-test-4"

Write-Host "`n[Turn 1] What does P0300 mean" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What does P0300 mean?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] What does P0301 mean" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What about P0301?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 3] How are those two codes related — needs context from both turns" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "How are those two codes related to each other?" `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 5: Three-turn hybrid — data + code + combined interpretation
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 5: Three-turn hybrid chain ==========" -ForegroundColor Cyan
$session = "rag-conv-test-5"

Write-Host "`n[Turn 1] PAL — catalyst temperature average" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What is the average catalyst temperature in drive1.csv?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] RAG — what does P0420 mean" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "What does fault code P0420 mean?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 3] Combined — could the temperature reading indicate P0420" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --docs-dir $docs `
  --question "Based on the catalyst temperature we saw, could that indicate a P0420 fault?" `
  --show-route `
  --show-details
