# Caching Bug Test Instructions

## Purpose
Verify that different images produce different result IDs (fix for issue where second upload showed first image's result).

## Prerequisites
1. Ensure server is running: `cd c:\Users\akash\pneumonia-detection && python -m src.app`
2. Have browser open to: http://localhost:8000/upload
3. Have two DIFFERENT chest X-ray images ready to upload

## Test Steps

### Step 1: Upload First Image
1. Open **Browser DevTools** (Press `F12`)
2. Go to **Console** tab
3. Click "Choose File" and select **first image**
4. Verify in console:
   - `File: [filename1.jpg]`
   - `Size: [size1] bytes`
   - `File Hash: [hash1|hash2|hash3]`
5. Click **Submit**
6. Watch console for:
   - `[FORM SUBMIT] File details for upload:`
   - `[FETCH] Response status: 200 OK`
   - `[JSON PARSE] Success:` with `redirect` URL
   - **NOTE THE RESULT ID in redirect URL** (format: `/result/[token]_[timestamp]`)
7. Result page loads

### Step 2: Return to Upload Page
1. Click **"Analyze Another Image"** button
2. Wait for page to reload
3. In console, verify:
   - `Form reset complete`
   - `fileInput.files.length: 0` (should be zero!)
   - `fileInput.value: ` (should be empty!)

### Step 3: Upload Second Image
1. Click "Choose File" and select **DIFFERENT image**
2. Verify in console:
   - `File: [filename2.jpg]` (DIFFERENT from first!)
   - `Size: [size2] bytes` (SHOULD DIFFER from size1)
   - `File Hash: [hash4|hash5|hash6]` (DIFFERENT from earlier hash!)
3. Click **Submit**
4. Watch console for:
   - `[FORM SUBMIT] File details for upload:`
   - `[FETCH] Response status: 200 OK`
   - `[JSON PARSE] Success:` with `redirect` URL
   - **NOTE THE RESULT ID** (should be COMPLETELY DIFFERENT from first!)
5. Result page loads

### Step 4: Verify Server Logs
In server terminal, you should see:

```
================================================================================
NEW UPLOAD REQUEST - Processing for user: ...
================================================================================
[FILE INFO] Name: [filename1.jpg]
[FILE INFO] Size: [size1] bytes
[FILE INFO] Hash (SHA256): [hash1]
...
[RESULT ID GENERATION]
Random token: [token1]
Timestamp (ms): [timestamp1]
Unique Key: [token1]_[timestamp1]
...

================================================================================
NEW UPLOAD REQUEST - Processing for user: ...
================================================================================
[FILE INFO] Name: [filename2.jpg]
[FILE INFO] Size: [size2] bytes
[FILE INFO] Hash (SHA256): [hash2]  <-- SHOULD BE DIFFERENT!
...
[RESULT ID GENERATION]
Random token: [token2]  <-- SHOULD BE DIFFERENT!
Timestamp (ms): [timestamp2]  <-- SHOULD BE DIFFERENT!
Unique Key: [token2]_[timestamp2]  <-- SHOULD BE COMPLETELY DIFFERENT!
```

## Success Criteria

✅ **PASS** if:
1. First image shows result ID: `/result/[token1]_[timestamp1]`
2. After returning to upload, console shows form reset and empty file input
3. Second image is selected with DIFFERENT filename/size
4. Second submission shows result ID: `/result/[token2]_[timestamp2]`
5. The two result IDs are **completely different**
6. Server logs show different file hashes for both uploads

❌ **FAIL** if:
1. Both result IDs are identical
2. File input still shows previous file after "Analyze Another Image"
3. File sizes are the same (indicates same file being submitted)
4. File hashes in server logs are identical

## Troubleshooting

### Same result ID appears for different images:
- Check if form reset is showing in console
- Check if fileInput.files.length is actually 0 after reset
- Verify DataTransfer API is being used: `fileInput.files = new DataTransfer().items`

### File sizes appear identical:
- Ensure you're using two COMPLETELY different images
- Check file hash - if identical, same file is being uploaded twice

### Server logs show same file hash:
- Browser may still have old file cached in input element
- Look at FormData contents in console before fetch
- The `[FORM SUBMIT] File details for upload:` should show different details

## Critical Logs to Watch

**Browser Console:**
```
[FORM SUBMIT] File details for upload:
  Name: 
  Size:  bytes
  Type: 
  Last Modified: 
  File Hash: |||

[FORM SUBMIT] FormData contents before fetch:
```

**Server Terminal:**
```
[FILE INFO] Name: 
[FILE INFO] Size:  bytes
[FILE INFO] Hash (SHA256): 

[RESULT ID GENERATION]
Random token: 
Timestamp (ms): 
Unique Key: _
```

These must be DIFFERENT for both uploads to confirm the bug is fixed.
