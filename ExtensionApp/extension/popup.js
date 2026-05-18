// Function to be injected into the active web page
function scrapeEmailContent() {
    const host = window.location.hostname.toLowerCase();

    const looksLikeEmailInbox = (
        host.includes('mail.google.com') ||
        host.includes('outlook.office.com') ||
        host.includes('outlook.live.com') ||
        host.includes('mail.yahoo.com')
    );

    if (!looksLikeEmailInbox) {
        return { ok: false, reason: 'not_email_inbox' };
    }

    // Only scan if an actual email message is open
    const gmailBody = document.querySelector('.a3s.aiL');
    const outlookBody = document.querySelector('[aria-label="Message body"]');
    const emailBody = gmailBody || outlookBody;

    if (!emailBody) {
        return { ok: false, reason: 'not_open_email' };
    }

    // Extract subject from page title or email header
    const subject = document.title || "";
    
    function normalizeSenderLabel(nameText, emailText) {
        const cleanedName = (nameText || '').trim();
        const cleanedEmail = (emailText || '').trim();

        if (cleanedName && cleanedName.toLowerCase() !== 'me') {
            return cleanedName;
        }

        if (cleanedEmail) {
            return cleanedEmail;
        }

        return "Unknown Sender";
    }

    // Get sender/from information
    let senderName = "";
    let senderEmail = "";
    
    // Gmail: Look for sender in header
    const gmailSenderName = document.querySelector('span[email]');
    if (gmailSenderName) {
        senderName = (gmailSenderName.textContent || "").trim();
        senderEmail = (gmailSenderName.getAttribute('email') || "").trim();

        // Gmail often shows "me" for your own account; prefer actual email in that case.
        if (senderName.toLowerCase() === 'me' && senderEmail) {
            senderName = "";
        }
    }
    
    // Outlook: Look for from field
    const outlookFromField = document.querySelector('[data-test-id="from-field"]');
    if (outlookFromField && !senderName) {
        senderName = outlookFromField.textContent || "";
    }
    
    // Fallback: extract from subject or first visible header element
    if (!senderName) {
        const headerElements = document.querySelectorAll('[role="heading"], .bAk, [data-tooltip-id]');
        for (let elem of headerElements) {
            const text = elem.textContent;
            if (text && text.includes('@')) {
                senderEmail = text;
                break;
            }
        }
    }
    
    const displaySender = normalizeSenderLabel(senderName, senderEmail);

    // Build header with sender information
    const headerText = `From: ${displaySender}\nSubject: ${subject}`;
    
    // Get complete email body text (all content, not just visible portion)
    const bodyText = emailBody.innerText || emailBody.textContent || "";

    // Combine all content, limit to 10000 characters to prevent overloading
    let combinedText = headerText + "\n\n" + bodyText;
    
    return { 
        ok: true, 
        text: combinedText.substring(0, 10000),
        sender: displaySender,
        subject: subject,
        bodyText: bodyText.substring(0, 8000)
    };
}

const MANUAL_DRAFT_KEY = 'phishing_detector_manual_draft';

function getSelectedModel() {
    const select = document.getElementById('modelSelect');
    return select ? select.value : 'distilbert';
}

function getManualInputs() {
    return {
        subject: document.getElementById('inputSubject'),
        header: document.getElementById('inputHeader'),
        body: document.getElementById('inputBody'),
    };
}

function buildManualText() {
    const { subject, header, body } = getManualInputs();
    const subjectText = subject.value.trim();
    const headerText = header.value.trim();
    const bodyText = body.value.trim();

    return [
        subjectText ? `Subject: ${subjectText}` : '',
        headerText ? `Header: ${headerText}` : '',
        bodyText ? `Body: ${bodyText}` : ''
    ].filter(Boolean).join('\n\n').trim();
}

function saveManualDraft() {
    const { subject, header, body } = getManualInputs();
    chrome.storage.local.set({
        [MANUAL_DRAFT_KEY]: {
            subject: subject.value,
            header: header.value,
            body: body.value,
        }
    });
}

function restoreManualDraft() {
    const { subject, header, body } = getManualInputs();
    chrome.storage.local.get([MANUAL_DRAFT_KEY], (result) => {
        const draft = result[MANUAL_DRAFT_KEY];
        if (!draft) {
            return;
        }

        subject.value = draft.subject || '';
        header.value = draft.header || '';
        body.value = draft.body || '';
    });
}

function initPopupPersistence() {
    restoreManualDraft();

    const { subject, header, body } = getManualInputs();
    [subject, header, body].forEach((input) => {
        input.addEventListener('input', saveManualDraft);
        input.addEventListener('change', saveManualDraft);
        input.addEventListener('blur', saveManualDraft);
        input.addEventListener('paste', () => {
            window.setTimeout(saveManualDraft, 0);
        });
    });

    window.addEventListener('beforeunload', saveManualDraft);
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            saveManualDraft();
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPopupPersistence);
} else {
    initPopupPersistence();
}

// Handle Auto-Scan button
document.getElementById('scanPageBtn').addEventListener('click', async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // Show loading state immediately for a seamless feel
        const loader = document.getElementById('loader');
        const resultBox = document.getElementById('resultBox');
        loader.classList.remove('hidden');
        resultBox.classList.add('hidden');

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: scrapeEmailContent,
        }, async (injectionResults) => {
            if (chrome.runtime.lastError || !injectionResults || !injectionResults.length) {
                alert('Cannot read this page. Try pasting text manually.');
                loader.classList.add('hidden');
                return;
            }

            const scanResult = injectionResults[0].result;
            if (!scanResult || scanResult.ok === false) {
                alert('Not email inbox. Open an email inbox page or use manual entry below.');
                loader.classList.add('hidden');
                return;
            }

            const scrapedText = scanResult.text;
            const scrapedSender = scanResult.sender || "Unknown";
            const scrapedSubject = scanResult.subject || "";
            const scrapedBody = scanResult.bodyText || "";
            
            if (scrapedText && scrapedText.trim().length > 0) {
                // Parse the scraped text and populate the fields
                const { subject, header, body } = getManualInputs();
                
                // Populate sender/From in header field
                subject.value = scrapedSubject;
                header.value = `From: ${scrapedSender}`;
                body.value = scrapedBody;
                
                // Save the populated fields
                saveManualDraft();
                
                // Automatically analyze the scanned content
                try {
                    const response = await fetch('http://127.0.0.1:5000/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: scrapedText, model: getSelectedModel() })
                    });

                    const data = await response.json();
                    const resultStatus = document.getElementById('resultStatus');
                    const confidenceLevel = document.getElementById('confidenceLevel');

                    if (response.ok) {
                        resultStatus.textContent = `Result: ${data.prediction}`;
                        resultStatus.style.color = data.prediction === 'BEC' ? '#d32f2f' : '#388e3c';
                        const modelTag = data.model_used ? ` | Model: ${data.model_used}` : '';
                        const fallbackTag = data.fallback_used ? ' | ⚠️ Fallback' : '';
                        const confidence = data.confidence !== undefined ? data.confidence : data.bec_probability;
                        const confidenceValue = typeof confidence === 'string' ? confidence : (confidence * 100).toFixed(1) + '%';
                        confidenceLevel.textContent = `BEC Probability: ${confidenceValue}${modelTag}${fallbackTag}`;
                        resultBox.classList.remove('hidden');
                    } else {
                        alert(data.error || 'Failed to analyze text.');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Could not connect to the backend server. Make sure the Flask api is running.');
                } finally {
                    loader.classList.add('hidden');
                }

            } else {
                alert('No content found on this page to scan.');
                loader.classList.add('hidden');
            }
        });
    } catch (err) {
        console.error('Error injecting script:', err);
        alert('Could not access the page content.');
    }
});

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = buildManualText();

    if (!text) {
        alert('Please enter at least one of Subject, Header, or Body to analyze.');
        return;
    }

    const loader = document.getElementById('loader');
    const resultBox = document.getElementById('resultBox');
    const resultStatus = document.getElementById('resultStatus');
    const confidenceLevel = document.getElementById('confidenceLevel');

    // Reset UI
    loader.classList.remove('hidden');
    resultBox.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text, model: getSelectedModel() })
        });

        const data = await response.json();
        
        if (response.ok) {
            resultStatus.textContent = `Result: ${data.prediction}`;
            resultStatus.style.color = data.prediction === 'BEC' ? '#d32f2f' : '#388e3c';
            const modelTag = data.model_used ? ` | Model: ${data.model_used}` : '';
            const fallbackTag = data.fallback_used ? ' | ⚠️ Fallback' : '';
            const confidence = data.confidence !== undefined ? data.confidence : data.bec_probability;
            const confidenceValue = typeof confidence === 'string' ? confidence : (confidence * 100).toFixed(1) + '%';
            confidenceLevel.textContent = `BEC Probability: ${confidenceValue}${modelTag}${fallbackTag}`;
            resultBox.classList.remove('hidden');
        } else {
            alert(data.error || 'Failed to analyze text.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the backend server. Make sure the Flask api is running.');
    } finally {
        loader.classList.add('hidden');
    }
});

// Handle Reset button
document.getElementById('resetBtn').addEventListener('click', () => {
    const { subject, header, body } = getManualInputs();
    
    // Clear all text fields
    subject.value = '';
    header.value = '';
    body.value = '';
    
    // Clear result box
    const resultBox = document.getElementById('resultBox');
    resultBox.classList.add('hidden');
    
    // Save the cleared state
    saveManualDraft();
});
