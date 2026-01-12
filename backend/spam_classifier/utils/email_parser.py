"""
Email Parser for Spam Detection
Based on the Parser class from the provided Jupyter notebooks
"""
import email
import string
from html.parser import HTMLParser
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class MLStripper(HTMLParser):
    """HTML parser to strip HTML tags from email content"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    """Remove HTML tags from text"""
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class Parser:
    """Email parser that extracts and processes email content"""
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse_text(self, text):
        """Parse raw email text content"""
        try:
            # Try to parse as email message
            msg = email.message_from_string(text)
            return self.get_email_content(msg)
        except Exception:
            # If not a proper email format, treat as plain text
            return {
                "subject": [],
                "body": self.tokenize(text),
                "content_type": "text/plain"
            }

    def get_email_content(self, msg):
        """Extract the email content."""
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(), msg.get_content_type())
        content_type = msg.get_content_type()
        
        return {
            "subject": subject,
            "body": body,
            "content_type": content_type
        }

    def get_email_body(self, payload, content_type):
        """Extract the body of the email."""
        body = []
        if isinstance(payload, str) and content_type == 'text/plain':
            return self.tokenize(payload)
        elif isinstance(payload, str) and content_type == 'text/html':
            return self.tokenize(strip_tags(payload))
        elif isinstance(payload, list):
            for p in payload:
                body += self.get_email_body(p.get_payload(),
                                            p.get_content_type())
        return body

    def tokenize(self, text):
        """
        Transform a text string into tokens.
        Clean the punctuation symbols and do stemming of the text.
        """
        if not text:
            return []
        
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        text = str(text)
        
        # Remove punctuation
        for c in self.punctuation:
            text = text.replace(c, " ")  # Replace with space instead of empty string
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        
        tokens = [w for w in text.split() if w.strip()]
        
        # Stemming of the tokens and remove stopwords
        stemmed_tokens = []
        for w in tokens:
            w_lower = w.lower()
            if w_lower not in self.stopwords and len(w_lower) > 1:  # Filter very short words
                stemmed_tokens.append(self.stemmer.stem(w_lower))
        
        return stemmed_tokens

    def get_text_from_tokens(self, parsed_email):
        """Convert parsed email tokens back to text for vectorization"""
        all_tokens = parsed_email['subject'] + parsed_email['body']
        return ' '.join(all_tokens)
