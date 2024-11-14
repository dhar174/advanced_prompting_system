# Personal Finance Management Application

class PersonalFinanceManager:
    def __init__(self):
        self.income = []
        self.expenses = []
        self.budget_limit = 0

    def set_budget(self, budget):
        self.budget_limit = budget
        print(f"Budget set to: {self.budget_limit}")

    def log_income(self, amount, category):
        self.income.append({'amount': amount, 'category': category})
        print(f"Income logged: {amount} in category {category}")

    def log_expense(self, amount, category):
        self.expenses.append({'amount': amount, 'category': category})
        print(f"Expense logged: {amount} in category {category}")

    def get_financial_summary(self):
        total_income = sum(item['amount'] for item in self.income)
        total_expenses = sum(item['amount'] for item in self.expenses)
        balance = total_income - total_expenses
        print(f"Total Income: {total_income}, Total Expenses: {total_expenses}, Balance: {balance}")

    def check_budget(self):
        total_expenses = sum(item['amount'] for item in self.expenses)
        if total_expenses > self.budget_limit:
            print("Alert: You are over your budget!")
        else:
            print("You are within your budget.")

# Example usage of the PersonalFinanceManager class:
if __name__ == "__main__":
    manager = PersonalFinanceManager()
    manager.set_budget(500)
    manager.log_income(1000, 'Salary')
    manager.log_expense(200, 'Groceries')
    manager.log_expense(350, 'Rent')
    manager.get_financial_summary()
    manager.check_budget()