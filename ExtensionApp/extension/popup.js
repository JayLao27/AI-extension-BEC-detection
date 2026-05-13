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

    // Only scan if an actual email message is open, not the inbox list.
    const gmailBody = document.querySelector('.a3s.aiL');
    const outlookBody = document.querySelector('[aria-label="Message body"]');
    const emailBody = gmailBody || outlookBody;

    if (!emailBody) {
        return { ok: false, reason: 'not_open_email' };
    }

    // Grab the page title, which often contains the email subject
    const subject = document.title || "";
    const bodyText = emailBody.innerText || "";

    // Combine subject and body, limit to 5000 characters to prevent overloading the model
    let combinedText = "Subject: " + subject + "\n\n" + bodyText;
    return { ok: true, text: combinedText.substring(0, 5000) };
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
            if (scrapedText && scrapedText.trim().length > 0) {
                
                // Do the analysis blindly in the background without populating the text box
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
                        resultStatus.style.color = data.prediction === 'Phishing' ? '#d32f2f' : '#388e3c';
                        const modelTag = data.model_used ? ` | Model: ${data.model_used}` : '';
                        const fallbackTag = data.fallback_used ? ' | Fallback' : '';
                        confidenceLevel.textContent = `Phishing Probability: ${(data.confidence * 100).toFixed(1)}%${modelTag}${fallbackTag}`;
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
            resultStatus.style.color = data.prediction === 'Phishing' ? '#d32f2f' : '#388e3c';
            const modelTag = data.model_used ? ` | Model: ${data.model_used}` : '';
            const fallbackTag = data.fallback_used ? ' | Fallback' : '';
            confidenceLevel.textContent = `Phishing Probability: ${(data.confidence * 100).toFixed(1)}%${modelTag}${fallbackTag}`;
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
