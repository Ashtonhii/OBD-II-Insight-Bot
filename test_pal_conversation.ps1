# test_pal_conversation.ps1
# Multi-turn PAL conversation tests — verifies that follow-up questions
# carry context from the previous answer within the same session.

$csv = "data/obdiidata/drive1.csv"

# -----------------------------------------------------------------------
# Test 1: Basic aggregate then comparison follow-up
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 1: Aggregate + comparison follow-up ==========" -ForegroundColor Cyan
$session = "pal-conv-test-1"

Write-Host "`n[Turn 1] Average RPM" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What is the average engine RPM in drive1.csv?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] Compare to max — should reference the average from Turn 1" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "How does that compare to the maximum RPM?" `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 2: Conditional filter then unit conversion follow-up
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 2: Filter then unit conversion follow-up ==========" -ForegroundColor Cyan
$session = "pal-conv-test-2"

Write-Host "`n[Turn 1] Mean speed while moving" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What is the mean vehicle speed when the car is moving?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] Convert to km/h — follow-up on the previous speed figure" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "Convert that to kilometres per hour." `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 3: Percentile then threshold follow-up
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 3: Percentile then threshold follow-up ==========" -ForegroundColor Cyan
$session = "pal-conv-test-3"

Write-Host "`n[Turn 1] 90th percentile RPM" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What is the 90th percentile of engine RPM in drive1.csv?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] How many rows exceed that threshold" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "How many readings exceed that threshold?" `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 4: Correlation then interpretation follow-up
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 4: Correlation then interpretation ==========" -ForegroundColor Cyan
$session = "pal-conv-test-4"

Write-Host "`n[Turn 1] Correlation between RPM and speed" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What is the Pearson correlation between engine RPM and vehicle speed in drive1.csv?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] Is that a strong correlation — requires knowing the number from Turn 1" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "Is that considered a strong correlation?" `
  --show-route `
  --show-details

# -----------------------------------------------------------------------
# Test 5: Three-turn chain — aggregate, filter, then std deviation
# -----------------------------------------------------------------------
Write-Host "`n========== TEST 5: Three-turn chain ==========" -ForegroundColor Cyan
$session = "pal-conv-test-5"

Write-Host "`n[Turn 1] Mean coolant temperature" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What is the average coolant temperature in drive1.csv?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 2] Percentage of time coolant was above that mean" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "What percentage of the time was the coolant temperature above that value?" `
  --show-route `
  --show-details

Write-Host "`n[Turn 3] Standard deviation of coolant temperature" -ForegroundColor Yellow
.\venv\Scripts\python.exe .\src\ask_agent.py `
  --session-id $session `
  --csv $csv `
  --question "Now what is the standard deviation of the coolant temperature?" `
  --show-route `
  --show-details
