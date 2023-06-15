from dataclasses import dataclass, field

@dataclass
class Prediction:
    theta_0: float = 0.0
    theta_1: float = 0.0

    def get_integer(self, prompt):
        while True:
            try:
                value = int(input(prompt))
                return value
            except:
                print("Invalid input. Please enter an integer.")
    
    def compute_prediction(self):
        x = self.get_integer("Enter the mileage of your car:")
        y_hat = self.theta_0 + self.theta_1 * x
        return y_hat

def main():
    pred = Prediction()
    prediction_result = pred.compute_prediction()
    if prediction_result >= 2:
        euros = "euros"
    else:
        euros = "euro"
    print(f"The predicted value is: {prediction_result} {euros}")

if __name__ == "__main__":
    main()
