class Res():
    def __init__(self, restaurant_name, cuisine_type):
        self.cuisine_type = cuisine_type
        self.restaurant_name = restaurant_name
        self.number_served = 0

    def describe_restaurant(self):
        print(self.restaurant_name.title() + ' ' + self.cuisine_type.title())

    def open_restaurant(self):
        print(self.restaurant_name.title() + ' is open!')

    def set_number_served(self, number_served):
        self.number_served = number_served

    def increment_number_served(self, inc):
        self.number_served += inc


class IceCreamStand(Res):
    def __init__(self, restaurant_name, cuisine_type, flavor):
        super().__init__(restaurant_name, cuisine_type)
        self.flavor = flavor

    def check_flavors(self):
        for i in self.flavor:
            print(i)