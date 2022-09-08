class Murtoluku:
    def __init__(self, a, b):
        self.osoittaja = a
        self.nimittaja = b

    def tulosta(self):
        print(f'{self.osoittaja}/{self.nimittaja}')
    
    def sievenna(self):
        s = self.syt()
        self.osoittaja //= s
        self.nimittaja //= s
    
    def syt(self):
        a = self.osoittaja
        b = self.nimittaja
        while b != 0:
             t = b
             b = a % b
             a = t
        return a
             
            
            

test = Murtoluku(34562,311058)

test.tulosta()

test.sievenna()

test.tulosta()
        
