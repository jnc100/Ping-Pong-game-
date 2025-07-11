import turtle
import time

# Set up the screen
win = turtle.Screen()
win.title("Ping Pong Game")
win.bgcolor("black")
win.setup(width=800, height=600)
win.tracer(0)

# Paddle Class
class Paddle(turtle.Turtle):
    def __init__(self, x_pos):
        super().__init__()
        self.shape("square")
        self.color("white")
        self.shapesize(stretch_wid=5, stretch_len=1)
        self.penup()
        self.goto(x_pos, 0)

    def go_up(self):
        if self.ycor() < 250:
            self.sety(self.ycor() + 20)

    def go_down(self):
        if self.ycor() > -240:
            self.sety(self.ycor() - 20)

# Ball Class
class Ball(turtle.Turtle):
    def __init__(self):
        super().__init__()
        self.shape("circle")
        self.color("red")
        self.penup()
        self.goto(0, 0)
        self.dx = 0.2
        self.dy = 0.2
        self.speed_multiplier = 1.01

    def move(self):
        self.setx(self.xcor() + self.dx)
        self.sety(self.ycor() + self.dy)

    def bounce_y(self):
        self.dy *= -1

    def bounce_x(self):
        self.dx *= -1
        self.dx *= self.speed_multiplier
        self.dy *= self.speed_multiplier

    def reset_position(self):
        self.goto(0, 0)
        self.dx = 0.2
        self.dy = 0.2
        self.bounce_x()

# Create paddles and ball
paddle_a = Paddle(-350)
paddle_b = Paddle(350)
ball = Ball()

# Keyboard bindings
win.listen()
win.onkeypress(paddle_a.go_up, "w")
win.onkeypress(paddle_a.go_down, "s")
win.onkeypress(paddle_b.go_up, "Up")
win.onkeypress(paddle_b.go_down, "Down")

# Game loop
while True:
    win.update()
    ball.move()
    time.sleep(0.01)

    # Border collision
    if ball.ycor() > 290 or ball.ycor() < -290:
        ball.bounce_y()

    # Right and left wall
    if ball.xcor() > 390:
        ball.reset_position()
    elif ball.xcor() < -390:
        ball.reset_position()

    # Paddle collision
    if (340 < ball.xcor() < 350 and paddle_b.ycor() - 50 < ball.ycor() < paddle_b.ycor() + 50) or \
       (-350 < ball.xcor() < -340 and paddle_a.ycor() - 50 < ball.ycor() < paddle_a.ycor() + 50):
        ball.bounce_x()
