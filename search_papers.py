import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def search_arxiv(query):
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': f'all:"{query}"',
        'start': 0,
        'max_results': 10,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"
    
    try:
        with urllib.request.urlopen(full_url, context=ctx) as response:
            if response.status == 200:
                data = response.read()
                root = ET.fromstring(data)
                print(f'\n--- Results for: {query} ---')
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
                    published = entry.find('{http://www.w3.org/2005/Atom}published').text
                    authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
                    link = entry.find('{http://www.w3.org/2005/Atom}id').text
                    print(f"Title: {title}")
                    print(f"Published: {published}")
                    print(f"Authors: {', '.join(authors)}")
                    print(f"URL: {link}")
                    print('-'*40)
            else:
                print('Error:', response.status)
    except Exception as e:
        print('Exception:', e)

search_arxiv('Automotive Ethernet Intrusion Detection')
search_arxiv('Hybrid Intrusion Detection CAN Ethernet')
search_arxiv('Lightweight Intrusion Detection CAN')
