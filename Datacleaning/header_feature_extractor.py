"""
Header-based Impersonation Classifier Module

Implements the paper's key insight: BEC detection requires a two-stage cascade:
1. Header-based impersonation gate (this module)
2. Body content classifier (DistilRoBERTa)

Key features from Cidon et al. Table 3:
- reply-to ≠ sender mismatch
- sender name vs email address mismatch (e.g., "John Smith" + "malicious@domain.com")
- historical sender frequency (how often this sender+email pair has been observed)
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict


class HeaderFeatureExtractor:
    """
    Extracts impersonation-based features from email headers.
    These features form the gating classifier before body content analysis.
    """
    
    def __init__(self):
        """Initialize the extractor and sender frequency database."""
        self.sender_frequency = defaultdict(int)  # (sender_name, sender_email) -> count
        self.reply_to_frequency = defaultdict(int)  # (sender_email, reply_to) -> count
        self.domain_whitelist = set()  # Trusted domains
        
    def _extract_email(self, text):
        """
        Extract email address from text (e.g., "John Smith <john@example.com>" -> "john@example.com")
        """
        if not isinstance(text, str):
            return ""
        
        match = re.search(r'<([^>]+)>', text)
        if match:
            return match.group(1).lower().strip()
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            return email_match.group(0).lower().strip()
        
        return text.lower().strip()
    
    def _extract_display_name(self, text):
        """
        Extract display name from text (e.g., "John Smith <john@example.com>" -> "John Smith")
        """
        if not isinstance(text, str):
            return ""
        
        match = re.search(r'^([^<]+)<', text)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _get_sender_domain(self, email):
        """Extract domain from email address."""
        if '@' in email:
            return email.split('@')[1].lower()
        return ""
    
    def _feature_reply_to_mismatch(self, sender_email, reply_to):
        """
        Feature 1: Is reply-to different from sender email?
        
        High-confidence impersonation signal: attacker changes reply-to to hide identity.
        Paper finding: 98.2% precision when combined with body classifier.
        
        Returns: 1 if mismatch (suspicious), 0 if match (legitimate)
        """
        if not sender_email or not reply_to:
            return 0
        
        sender_email = self._extract_email(sender_email)
        reply_to = self._extract_email(reply_to)
        
        return 1 if sender_email != reply_to else 0
    
    def _feature_name_email_mismatch(self, sender_name, sender_email):
        """
        Feature 2: Does sender name match email address?
        
        High-confidence impersonation signal: display name is CEO, but email is random domain.
        Example: "Robert Smith" + "urgent.payments@[random-domain].com"
        
        Returns: impersonation risk score (0-1)
        """
        if not sender_name or not sender_email:
            return 0
        
        sender_email = self._extract_email(sender_email)
        domain = self._get_sender_domain(sender_email)
        
        # Check if domain is a common mail provider (Gmail, Outlook, Yahoo)
        # Real CEOs rarely email from public mail providers during business
        common_providers = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'}
        if domain in common_providers:
            return 0.5  # Moderate risk - could be legitimate but unusual for executive
        
        # Check if name appears in email address (legitimate case)
        name_parts = sender_name.lower().split()
        email_local_part = sender_email.split('@')[0].lower()
        
        # If any part of name is in the email, it's likely legitimate
        for part in name_parts:
            if len(part) > 3 and part in email_local_part:
                return 0  # Name found in email, likely legitimate
        
        return 1  # Name and email are completely unrelated - suspicious
    
    def _feature_sender_frequency(self, sender_name, sender_email, is_training=False):
        """
        Feature 3: Historical sender frequency
        
        Core impersonation signal: spoofed sender+email combos are rare/first-time.
        Paper finding: Legitimate business pairs appear 5-100+ times; BEC attacks appear 1-2 times.
        
        Args:
            sender_name: Display name
            sender_email: Email address
            is_training: If True, update frequency counts; if False, just query
        
        Returns: sender_rarity_score (0-1, where 1 = never seen before, 0 = frequently seen)
        """
        sender_email = self._extract_email(sender_email)
        key = (sender_name.lower(), sender_email)
        
        if is_training:
            self.sender_frequency[key] += 1
        
        count = self.sender_frequency[key]
        
        # Rarity scoring: never seen = 1.0, seen 10+ times = 0.0
        # Formula: 1 / (1 + log(count))
        if count == 0:
            return 1.0  # Never seen
        elif count >= 10:
            return 0.0  # Frequently seen, legitimate
        else:
            return 1.0 / (1.0 + np.log(count + 1))
    
    def extract_header_features(self, sender, reply_to, subject, is_training=False):
        """
        Extract all impersonation features from email header.
        
        Args:
            sender: Raw "From" field (e.g., "John Smith <john@example.com>")
            reply_to: Raw "Reply-To" field (optional)
            subject: Email subject line (context)
            is_training: Whether to update sender frequency stats
        
        Returns: Dictionary of feature values and impersonation_score
        """
        if not isinstance(sender, str):
            sender = ""
        if not isinstance(reply_to, str):
            reply_to = ""
        if not isinstance(subject, str):
            subject = ""
        
        sender_email = self._extract_email(sender)
        sender_name = self._extract_display_name(sender) or sender_email.split('@')[0]
        reply_to_email = self._extract_email(reply_to) if reply_to else sender_email
        
        # Extract individual features
        reply_to_mismatch = self._feature_reply_to_mismatch(sender_email, reply_to_email)
        name_email_mismatch = self._feature_name_email_mismatch(sender_name, sender_email)
        sender_rarity = self._feature_sender_frequency(sender_name, sender_email, is_training)
        
        # Composite impersonation score (weighted average of features)
        # Paper weights: reply-to mismatch (high), name mismatch (high), sender rarity (medium)
        impersonation_score = (
            0.4 * reply_to_mismatch +
            0.35 * name_email_mismatch +
            0.25 * sender_rarity
        )
        
        features = {
            'sender_email': sender_email,
            'sender_name': sender_name,
            'reply_to_email': reply_to_email,
            'reply_to_mismatch': reply_to_mismatch,
            'name_email_mismatch': name_email_mismatch,
            'sender_rarity': sender_rarity,
            'impersonation_score': impersonation_score
        }
        
        return features
    
    def extract_features_batch(self, df, sender_col, reply_to_col, subject_col=None, is_training=False):
        """
        Extract header features for a batch of emails.
        
        Args:
            df: DataFrame with email data
            sender_col: Column name for sender/From field
            reply_to_col: Column name for reply-to field
            subject_col: Column name for subject (optional)
            is_training: Whether to update sender frequency stats
        
        Returns: DataFrame with extracted features
        """
        features_list = []
        
        for idx, row in df.iterrows():
            sender = row.get(sender_col, "")
            reply_to = row.get(reply_to_col, "")
            subject = row.get(subject_col, "") if subject_col else ""
            
            features = self.extract_header_features(sender, reply_to, subject, is_training)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        return df.join(features_df)
    
    def get_impersonation_gate(self, impersonation_score_threshold=0.5):
        """
        Returns a function that gates emails through the impersonation classifier.
        
        This implements the gating logic: only emails with impersonation_score >= threshold
        should proceed to the body classifier.
        
        Args:
            impersonation_score_threshold: Score above which to flag for body analysis
        
        Returns: Lambda function for filtering
        """
        return lambda score: score >= impersonation_score_threshold
    
    def get_sender_stats(self):
        """Return statistics on sender frequency distribution (for analysis)."""
        if not self.sender_frequency:
            return {}
        
        frequencies = list(self.sender_frequency.values())
        return {
            'total_unique_senders': len(self.sender_frequency),
            'median_frequency': np.median(frequencies),
            'mean_frequency': np.mean(frequencies),
            'max_frequency': max(frequencies),
            'min_frequency': min(frequencies)
        }


def apply_header_gate(df, extractor, impersonation_threshold=0.5):
    """
    Helper function: Apply the header impersonation gate to a dataframe.
    
    This implements the cascading architecture:
    1. Extract header features (this function)
    2. Filter by impersonation_score
    3. Only pass suspicious emails to body classifier
    
    Args:
        df: DataFrame with 'header' column (or raw email)
        extractor: HeaderFeatureExtractor instance
        impersonation_threshold: Score threshold for gating
    
    Returns: Tuple of (gated_df_high_risk, gated_df_low_risk)
    """
    # Parse header into components
    # For now, assuming 'header' column exists; adapt as needed
    
    features_df = extractor.extract_features_batch(df, 'sender', 'reply_to', 'subject')
    
    high_risk = features_df[features_df['impersonation_score'] >= impersonation_threshold]
    low_risk = features_df[features_df['impersonation_score'] < impersonation_threshold]
    
    return high_risk, low_risk
