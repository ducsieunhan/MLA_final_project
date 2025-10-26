print(("hello " * 3).strip(), ("python!" * 2).strip())

h = ["h", "e", "l", "l", "o"]
p = ["p", "y", "t", "h", "o", "n"]
print("".join(h), "".join(p))

hello = [
    "HH   HH EEEEEEE L       L        OOOOO ",
    "HH   HH EE      L       L       OO   OO",
    "HHHHHHH EEEE    L       L       OO   OO",
    "HH   HH EE      L       L       OO   OO",
    "HH   HH EEEEEEE LLLLLLL LLLLLLL  OOOOO ",
]

for line in hello:
    print(line)


title = ["hello", "python"]
body = ["learning", "by", "building", "strings"]
print(" ".join(title).title(), "-", " ".join(body))
