
import json
import csv
import datetime
import matplotlib.pyplot as plt

class PersonalFinanceManager:
    def __init__(self):
        self.transactions = []
        self.budgets = {}
        self.data_file = "finance_data.json"

    def load_data(self):
        """Loads financial data from a JSON file."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.transactions = data['transactions']
                self.budgets = data['budgets']
        except FileNotFoundError:
            print("No previous data found. Starting fresh.")
        except json.JSONDecodeError:
            print("Data file is corrupted, starting fresh.")
            self.transactions = []
            self.budgets = {}

    def save_data(self):
        """Saves financial data to a JSON file."""
        with open(self.data_file, 'w') as f:
            json.dump({'transactions': self.transactions, 'budgets': self.budgets}, f)

    def add_transaction(self, amount, category, label):
        """Logs an income or expense transaction."""
        if not self.validate_input(amount):
            print("Invalid input. Please enter a non-negative numeric value.")
            return
        
        transaction = {
            'date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'amount': amount,
            'category': category,
            'label': label
        }
        self.transactions.append(transaction)
        print(f"Transaction added: {transaction}")
        self.save_data()

    def set_budget(self, category, limit):
        """Sets a budget limit for a specific category."""
        if not self.validate_input(limit):
            print("Invalid budget value. Please enter a non-negative numeric value.")
            return
        
        self.budgets[category] = limit
        print(f"Budget set for {category}: {limit}")
        self.save_data()

    def check_budget(self):
        """Checks if any budgets are exceeded."""
        for transaction in self.transactions:
            category = transaction['category']
            if category in self.budgets:
                spent = sum(t['amount'] for t in self.transactions if t['category'] == category)
                if spent > self.budgets[category]:
                    print(f"Budget exceeded for {category}: Spent {spent}, Limit {self.budgets[category]}")

    def generate_summary(self):
        """Generates monthly and annual financial summaries."""
        monthly_summary = {}
        annual_summary = {}
        
        for transaction in self.transactions:
            month = transaction['date'][:7]  # Format YYYY-MM
            year = transaction['date'][:4]    # Format YYYY
            
            # Monthly summary
            if month not in monthly_summary:
                monthly_summary[month] = 0
            monthly_summary[month] += transaction['amount']
            
            # Annual summary
            if year not in annual_summary:
                annual_summary[year] = 0
            annual_summary[year] += transaction['amount']
        
        print("Monthly Summary:", monthly_summary)
        print("Annual Summary:", annual_summary)

    def visualize_data(self):
        """Visualizes financial data using matplotlib."""
        categories = {}
        
        for transaction in self.transactions:
            if transaction['category'] not in categories:
                categories[transaction['category']] = 0
            categories[transaction['category']] += transaction['amount']

        plt.bar(categories.keys(), categories.values())
        plt.xlabel('Categories')
        plt.ylabel('Amount')
        plt.title('Financial Overview by Categories')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def validate_input(value):
        """Validates that the input is a non-negative number."""
        return isinstance(value, (int, float)) and value >= 0

if __name__ == "__main__":
    pfm = PersonalFinanceManager()
    pfm.load_data()
    
    # Example usage
    pfm.add_transaction(500, 'Income', 'Job Salary')
    pfm.add_transaction(-100, 'Expense', 'Groceries')
    pfm.set_budget('Expense', 300)
    pfm.check_budget()
    pfm.generate_summary()
    pfm.visualize_data()
