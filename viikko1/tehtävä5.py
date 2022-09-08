import random

a_Nums = []
b_Nums = []
total = []
c_Answers = []

for i in range(0,5):
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    a_Nums.append(a)
    b_Nums.append(b)
    total.append(a*b)
    print (f'{a} * {b} = ')
    c = int(input())
    c_Answers.append(c)
    
for i in range(0,5):
    print(f'{a_Nums[i]} * {b_Nums[i]} = {c_Answers[i]}')
    if c_Answers[i] == total[i]:
        print("Oikein :)")
    else:
        print(f'Väärin :( Oikea vastaus: {a_Nums[i]} * {b_Nums[i]} = {total[i]}')




