

class Photometry:
    def __init__(self, band, flux, flux_err):
        self.band = band
        self.flux = flux
        self.flux_err = flux_err

    def __repr__(self):
        return f"{self.band}: {self.flux} +/- {self.flux_err}"
    
    def __str__(self):
        return f"{self.band}: {self.flux} +/- {self.flux_err}"
    
    def __add__(self, other):
        if isinstance(other, Photometry):
            return self.flux + other.flux
        elif isinstance(other, (int, float)):
            return self.flux + other
        else:
            raise TypeError(f"Cannot add {type(other)} to Photometry object")
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Photometry):
            return self.flux - other.flux
        elif isinstance(other, (int, float)):
            return self.flux - other
        else:
            raise TypeError(f"Cannot subtract {type(other)} from Photometry object")
        
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, Photometry):
            return self.flux * other.flux
        elif isinstance(other, (int, float)):
            return self.flux * other
        else:
            raise TypeError(f"Cannot multiply {type(other)} by Photometry object")
        
    def errs_in_quad(self, other):
        if isinstance(other, Photometry):
            return (self.flux_err**2 + other.flux_err**2)**0.5
        elif isinstance(other, (int, float)):
            return (self.flux_err**2 + other**2)**0.5
        else:
            raise TypeError(f"Cannot add {type(other)} to Photometry object")
        
def main():
    # make tweo photometry objects with the same bands and add
    p1 = Photometry("F277W", 1.0, 0.1)
    p2 = Photometry("F277W", 2.0, 0.2)
    p3 = p1 * p2
    print(p1.errs_in_quad(p2))
    print(p3)

if __name__ == "__main__":
    main()