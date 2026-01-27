"""
datasets.py - Dataset generators for Phase 6 testing

Datasets:
- Bash-100: Real bash commands
- Math-1K: Generated math equations
- Facts-200: Real-world Q&A facts
- Code-50: Python code snippets
"""

import random


def generate_bash_100():
    """100 real bash commands for testing."""
    commands = [
        # File operations
        ("list files", "ls -la"),
        ("show current directory", "pwd"),
        ("change to home", "cd ~"),
        ("make directory", "mkdir -p"),
        ("remove file", "rm -f"),
        ("copy file", "cp source dest"),
        ("move file", "mv source dest"),
        ("find files", "find . -name"),
        ("search in files", "grep -r"),
        ("file permissions", "chmod 755"),
        
        # Text processing
        ("count lines", "wc -l"),
        ("first 10 lines", "head -n 10"),
        ("last 10 lines", "tail -n 10"),
        ("sort file", "sort"),
        ("unique lines", "uniq"),
        ("replace text", "sed 's/old/new/g'"),
        ("extract column", "awk '{print $1}'"),
        ("concatenate files", "cat file1 file2"),
        
        # Process management
        ("list processes", "ps aux"),
        ("kill process", "kill -9"),
        ("background job", "command &"),
        ("check disk", "df -h"),
        ("check memory", "free -m"),
        ("system info", "uname -a"),
        
        # Network
        ("check connection", "ping"),
        ("download file", "curl -O"),
        ("list ports", "netstat -tlnp"),
        ("ssh connect", "ssh user@host"),
        ("scp file", "scp file user@host:"),
        
        # Git
        ("git status", "git status"),
        ("git add all", "git add ."),
        ("git commit", "git commit -m"),
        ("git push", "git push origin"),
        ("git pull", "git pull"),
        ("git branch", "git branch"),
        ("git checkout", "git checkout"),
        ("git log", "git log --oneline"),
        ("git diff", "git diff"),
        ("git clone", "git clone"),
        
        # Docker
        ("docker list", "docker ps"),
        ("docker images", "docker images"),
        ("docker run", "docker run -d"),
        ("docker stop", "docker stop"),
        ("docker logs", "docker logs"),
        ("docker exec", "docker exec -it"),
        ("docker build", "docker build -t"),
        
        # Package management
        ("install package", "apt install"),
        ("update packages", "apt update"),
        ("pip install", "pip install"),
        ("npm install", "npm install"),
        
        # More commands...
    ]
    
    # Expand to 100
    while len(commands) < 100:
        base = random.choice(commands[:50])
        commands.append(base)
    
    return [(f"Q: {q}? A:", f" {a}") for q, a in commands[:100]]


def generate_math_1k():
    """1000 math equations."""
    data = []
    
    # Addition (300)
    for _ in range(300):
        a, b = random.randint(1, 99), random.randint(1, 99)
        data.append((f"Q: {a}+{b}? A:", f" {a+b}"))
    
    # Multiplication (300)
    for _ in range(300):
        a, b = random.randint(2, 20), random.randint(2, 20)
        data.append((f"Q: {a}*{b}? A:", f" {a*b}"))
    
    # Subtraction (200)
    for _ in range(200):
        a, b = random.randint(20, 99), random.randint(1, 19)
        data.append((f"Q: {a}-{b}? A:", f" {a-b}"))
    
    # Division (200)
    for _ in range(200):
        b = random.randint(2, 10)
        a = b * random.randint(2, 10)
        data.append((f"Q: {a}/{b}? A:", f" {a//b}"))
    
    return data


def generate_facts_200():
    """200 real-world Q&A facts."""
    facts = [
        # Geography
        ("Capital of France", "Paris"),
        ("Capital of Japan", "Tokyo"),
        ("Capital of Germany", "Berlin"),
        ("Capital of Italy", "Rome"),
        ("Capital of Spain", "Madrid"),
        ("Capital of UK", "London"),
        ("Capital of USA", "Washington DC"),
        ("Capital of China", "Beijing"),
        ("Capital of India", "New Delhi"),
        ("Capital of Brazil", "Brasilia"),
        
        # Science
        ("Water formula", "H2O"),
        ("Salt formula", "NaCl"),
        ("Speed of light", "299792458 m/s"),
        ("Earth gravity", "9.8 m/s2"),
        ("Absolute zero", "-273.15 C"),
        ("Boiling point of water", "100 C"),
        ("Freezing point of water", "0 C"),
        
        # Tech
        ("Python creator", "Guido van Rossum"),
        ("Linux creator", "Linus Torvalds"),
        ("JavaScript creator", "Brendan Eich"),
        ("Apple founder", "Steve Jobs"),
        ("Microsoft founder", "Bill Gates"),
        ("Amazon founder", "Jeff Bezos"),
        ("Tesla CEO", "Elon Musk"),
        
        # Math
        ("Pi first 5 digits", "3.1415"),
        ("Euler number", "2.718"),
        ("Square root of 2", "1.414"),
        ("Binary of 10", "1010"),
        ("Hex of 255", "FF"),
        
        # General
        ("Largest planet", "Jupiter"),
        ("Smallest planet", "Mercury"),
        ("Earth star", "Sun"),
        ("Moon count Earth", "1"),
        ("Days in year", "365"),
        ("Months in year", "12"),
        ("Hours in day", "24"),
        ("Minutes in hour", "60"),
        ("Seconds in minute", "60"),
        
        # Colors
        ("Sky color", "Blue"),
        ("Grass color", "Green"),
        ("Blood color", "Red"),
        ("Sun color", "Yellow"),
        ("Snow color", "White"),
        
        # Opposites
        ("Opposite of hot", "Cold"),
        ("Opposite of big", "Small"),
        ("Opposite of fast", "Slow"),
        ("Opposite of up", "Down"),
        ("Opposite of left", "Right"),
    ]
    
    # Expand to 200 with variations
    expanded = []
    for q, a in facts:
        expanded.append((f"Q: {q}? A:", f" {a}"))
        expanded.append((f"Q: What is {q.lower()}? A:", f" {a}"))
        expanded.append((f"Q: Tell me {q.lower()} A:", f" {a}"))
        expanded.append((f"{q}?", f" {a}"))
    
    return expanded[:200]


def generate_code_50():
    """50 Python code snippets."""
    snippets = [
        # Functions
        ("def hello(): print('Hello')", "function to print hello"),
        ("def add(a,b): return a+b", "function to add two numbers"),
        ("def square(n): return n*n", "function to square a number"),
        ("def is_even(n): return n%2==0", "function to check if even"),
        ("def factorial(n): return 1 if n<=1 else n*factorial(n-1)", "factorial"),
        
        # List operations
        ("sorted(list)", "sort a list"),
        ("reversed(list)", "reverse a list"),
        ("sum(list)", "sum of list"),
        ("len(list)", "length of list"),
        ("max(list)", "max of list"),
        ("min(list)", "min of list"),
        
        # String operations
        ("s.upper()", "uppercase string"),
        ("s.lower()", "lowercase string"),
        ("s.strip()", "trim whitespace"),
        ("s.split()", "split string"),
        ("'.join(list)", "join list to string"),
        
        # File operations
        ("open('file.txt','r').read()", "read file"),
        ("open('file.txt','w').write(s)", "write file"),
        ("with open('f.txt') as f: data=f.read()", "read with context"),
        
        # Loops
        ("for i in range(10): print(i)", "loop 0 to 9"),
        ("while True: break", "infinite loop with break"),
        ("[x*2 for x in range(5)]", "list comprehension"),
        
        # Conditionals
        ("if x>0: print('positive')", "check positive"),
        ("x if condition else y", "ternary operator"),
        
        # Data structures
        ("{'key': 'value'}", "create dictionary"),
        ("dict.get('key', default)", "get with default"),
        ("set([1,2,2,3])", "create set"),
    ]
    
    return [(f"Python: {desc}? A:", f" {code}") for code, desc in snippets[:50]]


def get_all_datasets():
    """Get all datasets combined."""
    return {
        'bash': generate_bash_100(),
        'math': generate_math_1k(),
        'facts': generate_facts_200(),
        'code': generate_code_50(),
    }


if __name__ == "__main__":
    datasets = get_all_datasets()
    for name, data in datasets.items():
        print(f"{name}: {len(data)} samples")
