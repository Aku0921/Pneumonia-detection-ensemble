// JavaScript for Pneumonia Detection App

document.addEventListener('DOMContentLoaded', function() {
    // Add any interactive functionality here
    console.log('Pneumonia Detection App Loaded');
});

// Form validation helper
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (form) {
        return form.reportValidity();
    }
    return false;
}

// Image preview helper
function previewImage(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    
    if (input && input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            if (preview) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// Show loading spinner
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'block';
    }
}

// Hide loading spinner
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

// Copy to clipboard helper
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        console.log('Copied to clipboard');
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
    });
}
