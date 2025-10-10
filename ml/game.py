import pygame
import sys
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Pygame Example")

WHITE = (255, 255, 255)
BLUE = (0, 100, 255)

ball_radius = 20
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = 3, 2

clock = pygame.time.Clock()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

  
    ball_x += ball_speed_x
    ball_y += ball_speed_y

  
    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= WIDTH:
        ball_speed_x = -ball_speed_x
    if ball_y - ball_radius <= 0 or ball_y + ball_radius >= HEIGHT:
        ball_speed_y = -ball_speed_y

    screen.fill(WHITE)
    pygame.draw.circle(screen, BLUE, (ball_x, ball_y), ball_radius)
    pygame.display.flip()

    clock.tick(60)

pygame.quit()
sys.exit()
