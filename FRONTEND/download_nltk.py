import ssl
import certifi

print("Path to certificates:", certifi.where())

# If the above doesn't work, try the manual method
import urllib.request

def fix_ssl_certificates():
    try:
        # Attempt to download certificates manually
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Try downloading NLTK data without verification
        import nltk
        nltk.download('averaged_perceptron_tagger', download_dir=None)
        nltk.download('averaged_perceptron_tagger_eng', download_dir=None)
        
        print("NLTK downloads successful!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please try the manual certificate installation method.")

# Run the fix
fix_ssl_certificates()

# If this doesn't work, you may need to run these commands in Terminal
print("\nIf the above fails, run these commands in Terminal:")
print("/Applications/Python*/Install\ Certificates.command")