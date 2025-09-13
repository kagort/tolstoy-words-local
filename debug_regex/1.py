import re

pattern = "(вонями|(?:^|\s+|[!\"#$%&'()*+,-./:;<=>?@\[\]\\^_`{|}~]+)вонь(?:$|\s+|[!\"#$%&'()*+,-./:;<=>?@\[\]\\^_`{|}~]+)| вонях| воням| воней| вонью| вони)"
p = re.compile(pattern)

data = open("debug_regex/cleaned.txt", "rt", encoding="utf-8").read()

s = p.split(data)

print(f"len= {len(s)}")
print(pattern)

#print(s[2])
#print(s)
x = []
for i in range(len(s)):
    if i % 2 != 0:
        x.append(s[i])

print(len(x))
print(x)

forms = ['вонями','вонь','вонях', 'воням', 'воней', 'вонью', 'вони']

w = '*вонь'

punctuation = '\s+|[!\"#\$%&\'()\*\+,-\./:;<=>?@\[\]\\\^_`{|}~]+'
start_punctuation = r'(?:^|' + punctuation + ')'
end_punctuation = r'(?:$|' + punctuation + ')'
pattern2 = r'(?:' + '|'.join(start_punctuation + re.escape(w) + end_punctuation for w in forms) + r')'
print(pattern2)


pattern3 = r'(' + '|'.join(start_punctuation + re.escape(w) + end_punctuation for w in forms) + r')'

p2 = re.compile(pattern3)
print(p2.split(w))

s2 = p2.split(data)

print(f"len= {len(s2)}")
print(pattern3)

x2 = []
for i in range(len(s2)):
    if i % 2 != 0:
        x2.append(s2[i])

print(len(x2))
print(x2)

print(p2.split(w))