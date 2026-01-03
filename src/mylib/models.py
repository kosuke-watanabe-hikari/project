#テスト数理モデル
def test_f(x, y, params):
  a, b, c, d = params
  return a * x - b * x * y

def test_g(x, y, params):
  a, b, c, d = params
  return c * x * y - d * y