
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

class PersonalFinanceManager:
    def __init__(self):
        self.data_file = "finance_data.json"
        self.expenses = {}
        self.income = {}
        self.budgets = {}
        self.load_data()

    def load_data(self):
        """Load existing financial data from a file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as file:
                data = json.load(file)
                self.expenses = data.get('expenses', {})
                self.income = data.get('income', {})
                self.budgets = data.get('budgets', {})
    
    def save_data(self):
        """Save financial data to a file."""
        data = {
            'expenses': self.expenses,
            'income': self.income,
            'budgets': self.budgets
        }
        with open(self.data_file, 'w') as file:
            json.dump(data, file)

    def add_income(self, amount, source):
        """Add income to the record."""
        if amount < 0:
            print("Income amount must be positive.")
            return
        self.income[source] = self.income.get(source, 0) + amount
        self.save_data()
        print(f"Added income: {amount} from {source}.")

    def add_expense(self, amount, category):
        """Add expense to the record."""
        if amount < 0:
            print("Expense amount must be positive.")
            return
        self.expenses[category] = self.expenses.get(category, 0) + amount
        if category in self.budgets and self.expenses[category] > self.budgets[category]:
            print(f"Warning: You have exceeded your budget for {category}!")
        self.save_data()
        print(f"Added expense: {amount} for {category}.")

    def set_budget(self, category, budget_limit):
        """Set a budget limit for a specific category."""
        if budget_limit < 0:
            print("Budget limit must be non-negative.")
            return
        self.budgets[category] = budget_limit
        self.save_data()
        print(f"Set budget for {category} to {budget_limit}.")

    def generate_summary(self):
        """Generate monthly financial summary."""
        total_income = sum(self.income.values())
        total_expenses = sum(self.expenses.values())
        balance = total_income - total_expenses
        print("Financial Summary:")
        print(f"Total Income: {total_income}")
        print(f"Total Expenses: {total_expenses}")
        print(f"Balance: {balance}")

    def visualize_expenses(self):
        """Create a bar chart of expenses by category."""
        categories = list(self.expenses.keys())
        amounts = list(self.expenses.values())

        plt.bar(categories, amounts, color='blue')
        plt.xlabel('Categories')
        plt.ylabel('Expense Amounts')
        plt.title('Expenses by Category')
        plt.show()

    def backup_data(self, backup_file='finance_backup.json'):
        """Backup financial data to a specified file."""
        with open(backup_file, 'w') as file:
            json.dump({
                'expenses': self.expenses,
                'income': self.income,
                'budgets': self.budgets
            }, file)
        print(f"Data backed up to {backup_file}.")

    def restore_data(self, backup_file='finance_backup.json'):
        """Restore financial data from a backup file."""
        if os.path.exists(backup_file):
            with open(backup_file, 'r') as file:
                data = json.load(file)
                self.expenses = data.get('expenses', {})
                self.income = data.get('income', {})
                self.budgets = data.get('budgets', {})
                self.save_data()
            print(f"Data restored from {backup_file}.")
        else:
            print(f"No backup file found at {backup_file}.")

if __name__ == "__main__":
    manager = PersonalFinanceManager()
    while True:
        print("\nOptions:")
        print("1. Add Income")
        print("2. Add Expense")
        print("3. Set Budget")
        print("4. Generate Summary")
        print("5. Visualize Expenses")
        print("6. Backup Data")
        print("7. Restore Data")
        print("8. Exit")
        choice = input("Choose an option: ")

        try:
            if choice == '1':
                amount = float(input("Enter income amount: "))
                source = input("Enter source of income: ")
                manager.add_income(amount, source)
            elif choice == '2':
                amount = float(input("Enter expense amount: "))
                category = input("Enter expense category: ")
                manager.add_expense(amount, category)
            elif choice == '3':
                category = input("Enter budget category: ")
                budget_limit = float(input("Enter budget limit: "))
                manager.set_budget(category, budget_limit)
            elif choice == '4':
                manager.generate_summary()
            elif choice == '5':
                manager.visualize_expenses()
            elif choice == '6':
                backup_file = input("Enter backup file name: ")
                manager.backup_data(backup_file)
            elif choice == '7':
                backup_file = input("Enter backup file name to restore: ")
                manager.restore_data(backup_file)
            elif choice == '8':
                print("Exiting the application.")
                break
            else:
                print("Invalid option. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numbers where expected.")
