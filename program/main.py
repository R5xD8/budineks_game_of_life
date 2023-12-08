import tqdm
import GOL

s = 1000

game = GOL.almost(4*s, 4*s)

print(game.board)
game.step()

for i in tqdm.tqdm(range(1000)):
    game.step()

print(game.board)