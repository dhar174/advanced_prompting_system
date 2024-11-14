# Personal Finance Manager

import json
import matplotlib.pyplot as plt

class FinanceManager:
    def __init__(self):
        self.income = {}
        self.expenses = {}
        self.budget_limits = {}
        self.backup_file = 'backup.json'

    def add_income(self, category, amount):
        if category in self.income:
            self.income[category] += amount
        else:
            self.income[category] = amount

    def add_expense(self, category, amount):
        if category in self.expenses:
            self.expenses[category] += amount
        else:
            self.expenses[category] = amount
        
        if category in self.budget_limits and self.expenses[category] > self.budget_limits[category]:
            print(f"Warning: You have exceeded your budget for {category}")

    def set_budget(self, category, limit):
        self.budget_limits[category] = limit

    def generate_summary(self, period='monthly'):
        if period == 'monthly':
            print("Monthly Summary (Simplified):")
            for category, amount in self.expenses.items():
                print(f"{category}: {amount}")
                
        elif period == 'annual':
            print("Annual Summary (Simplified):")
            for category, amount in self.expenses.items():
                print(f"{category}: {amount * 12}")

    def visualize_expenses(self):
        categories = list(self.expenses.keys())
        amounts = list(self.expenses.values())
        plt.bar(categories, amounts)
        plt.xlabel('Category')
        plt.ylabel('Amount')
        plt.title('Expenses by Category')
        plt.show()

    def backup(self):
        try:
            with open(self.backup_file, 'w') as file:
                json.dump({'income': self.income, 'expenses': self.expenses}, file)
                print("Backup completed successfully.")
        except Exception as e:
            print(f"Failed to backup data: {e}")

    def restore(self):
        try:
            with open(self.backup_file, 'r') as file:
                data = json.load(file)
                self.income = data.get('income', {})
                self.expenses = data.get('expenses', {})
                print("Data restored successfully.")
        except Exception as e:
            print(f"Failed to restore data: {e}")

    def test_cases(self):
        # Basic test cases implemented for core functionalities
        self.add_income('Salary', 5000)
        self.add_expense('Food', 200)
        self.add_expense('Entertainment', 150)
        self.set_budget('Food', 250)
        self.generate_summary()
        self.visualize_expenses()
        self.backup()
        self.restore()

# Usage
fm = FinanceManager()
fm.test_cases()

# Note: Full implementation requires testing for additional edge cases and comprehensive documentation throughout.