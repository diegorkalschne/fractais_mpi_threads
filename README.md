# Fractais
Código Python que realiza a criação de imagens sobre fractais, mais especificadamente sobre o conjunto
de Julia e Mandelbrot. </br>
Utiliza MPI e paralelização com threads para as gerações das imagens.

- Endpoint disponível para visualização das imagens: </br>
http://localhost:5000/process?type=mandelbrot&xmax=1.5&xmin=-1.5&ymax=2&ymin=-1&interactions=100&zoom=1&real_part=-0.8&imag_part=0.156

## Dependências
- `pip install -r ./requirements.txt`
