from restaurant import *

class User():

    def __init__(self, first_name, last_name, phone, birthday):
        self.first_name = first_name
        self.last_name = last_name
        self.phone = phone
        self.birthday = birthday
        self.login_attempts = 0

    def describe_user(self):
        print('First name: ' + self.first_name.title() +
              '\nLast name: ' + self.last_name.title() +
              '\nPhone: ' + self.phone.title() +
              '\nBirthday: ' + self.birthday.title())

    def greet_user(self):
        print('Hello, ' + self.first_name.title())

    def increment_login_attempts(self):
        self.login_attempts += 1

    def reset_login_attempts(self):
        self.login_attempts = 0

class Privileges():

    def __init__(self):
        self.privileges = ['Permit to add messages', 'Permit to delete users', 'Permit to ban users']

    def show_privileges(self):
        for i in self.privileges:
            print(i)

class Admin(User):

    def __init__(self, first_name, last_name, phone, birthday):
        super().__init__(first_name, last_name, phone, birthday)
        self.privileges = Privileges()

my_res = Res('Astoria', 'russian')
iam = User('Fedor', 'Erastov', '89152260738', '25.12.1996')
my_ice_cream = IceCreamStand('Rose', 'Ice cream', ['Strawberry', 'Raspberry', 'Chocolate'])
my_admin = Admin('fedor', 'erastov', '89152260738', '25.12.1996')

my_admin.privileges.show_privileges()
