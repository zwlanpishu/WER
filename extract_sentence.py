with open("test_set_en_long.csv", "r+") as f:
    lines = [line.split("|")[1] for line in f]

with open("ref_trans_long.txt", "w+") as f:
    for line in lines:
        f.write(line + "\n")
