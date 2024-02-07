import kivy
kivy.require('1.0.6')
from kivy.app import App
from kivy.uix.widget import Widget 
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle, Color
from kivy.properties import ObjectProperty

class Calculator(BoxLayout):
    nse = ObjectProperty(None)
    bse = ObjectProperty(None)
    input_filter = ObjectProperty(None, allownone=True)

    def backward(self, express):
        pass

    def calculate(self, express):
        if not express: return
        try:
            self.display.text = str(eval(express))
        except Exception:
            self.display.text = 'error'

    def calc(self, text):
        print(text)

    def change(self):
        if self.nse.active:
            print('NSE')
        elif self.bse.active:
            print('BSE')

class CalculatorApp(App):
    def build(self):
        return Calculator()


if __name__ in ('__main__', '__android__'):
    CalculatorApp().run()