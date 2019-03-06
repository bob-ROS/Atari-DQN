import csv

with open("Output.txt") as f:
    reader = csv.reader(f, delimiter=",")
    count=0
    for row in reader:
        count=0
        for col in row:
            if count==0:
                print(col)
                fir = col
            if count==1:
                print(col)
                sec = col
            if count ==2:
                print(col)
                thi = col

            count += 1
     
    #first, sec, third = zip(for k,m,n in reader.items())



#print("{},{},{}".format(first,sec,third))
