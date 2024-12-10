from flask import Flask, request, send_file
from mpi4py import MPI
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from io import BytesIO

# Inicializa o MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# API Flask
app = Flask(__name__)

def mpi_worker():
    # rank 1, processa as mensagens
    if rank == 1:
        while True:
            data = comm.recv(source=0, tag=99)
            if data is None:
                break

            # Recupera os parâmetros
            xmin, xmax, ymin, ymax, res, max_iter, fractal_type, constant = data

            # Número de threads para calcular o fractal
            num_threads = 2

            # Realiza o cálculo do fractal
            output = calc_fractal_lines_threaded(xmin, xmax, ymin, ymax, res, max_iter, fractal_type, constant, num_threads=num_threads)

            # Envia o resultado de volta para rank 0
            comm.send(output, dest=0, tag=99)

@app.route('/process', methods=['GET'])
def process_request():
    print('opa')
    if rank == 0:
        # Parâmetros padrão
        res = 1000
        xmin = float(request.args.get('xmin', '-2'))
        xmax = float(request.args.get('xmax', '1'))
        ymin = float(request.args.get('ymin', '-1.5'))
        ymax = float(request.args.get('ymax', '1.5'))
        max_iter = request.args.get('interactions', 100, type=int)
        zoom_factor = float(request.args.get('zoom', '1'))
        fractal_type = request.args.get('type', 'julia')
        constant = None

        # Parâmetros padrão para Julia
        real_part = float(request.args.get('real_part', '-0.8'))
        imag_part = float(request.args.get('imag_part', '0.156'))

        if fractal_type == "julia":
            constant = complex(real_part, imag_part)

        # Envia as informações para todos os processos do MPI
        fractal_type = comm.bcast(fractal_type, root=0)
        constant = comm.bcast(constant, root=0)

        # Recalcula os parâmetros conforme o zoom fornecido
        if zoom_factor != 1:
            xmin, xmax, ymin, ymax = zoom_fractal(xmin, xmax, ymin, ymax, zoom_factor, 0, 0)

        # Envia os parâmetros de zoom e cálculo para o rank 1
        comm.send((xmin, xmax, ymin, ymax, res, max_iter, fractal_type, constant), dest=1, tag=99)

        # Espera receber a resposta do rank 1
        output = comm.recv(source=1, tag=99)

        # Cria a imagem do fractal utilizando matplotlib
        image_bytes = plot_fractal(output, xmin, xmax, ymin, ymax)

        # Cria um objeto BytesIO para enviar como resposta na requisição
        img_io = BytesIO(image_bytes)

        return send_file(img_io, mimetype='image/png')


# Função para calcular os limites do zoom
def zoom_fractal(xmin, xmax, ymin, ymax, zoom_factor, center_x, center_y):
    x_range = (xmax - xmin) / zoom_factor
    y_range = (ymax - ymin) / zoom_factor
    xmin = center_x - x_range / 2
    xmax = center_x + x_range / 2
    ymin = center_y - y_range / 2
    ymax = center_y + y_range / 2
    return xmin, xmax, ymin, ymax

# Função para plotar e salvar o fractal
def plot_fractal(output, xmin, xmax, ymin, ymax):
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(output, extent=(xmin, xmax, ymin, ymax), cmap='twilight', origin='lower')
        plt.colorbar(label="Iterações")
        plt.xlabel("Re")
        plt.ylabel("Im")

        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
        img_bytes.seek(0)

        # Libera os recursos utilizados para a criação da imagem
        plt.close()

        # Retorna os bytes da imagem
        return img_bytes.getvalue()
    except Exception as ex:
        print(f'Erro: {ex}')

# Função para calcular iterações
def fractal(c, z_initial, max_iter):
    z = z_initial
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return max_iter

# Função para calcular uma faixa de linhas do fractal, com threads
def calc_fractal_lines_threaded(xmin, xmax, ymin, ymax, res, max_iter, fractal_type, constant, num_threads=4):
    output = np.zeros((res, res), dtype=np.int32)
    real_vals = np.linspace(xmin, xmax, res)
    imag_vals = np.linspace(ymin, ymax, res)

    def calc_line(y_idx, y):
        line_output = np.zeros(res, dtype=np.int32)
        for x in range(res):
            if fractal_type == "mandelbrot":
                c = complex(real_vals[x], imag_vals[y])
                z_initial = 0
            elif fractal_type == "julia":
                c = constant
                z_initial = complex(real_vals[x], imag_vals[y])
            else:
                raise ValueError("Tipo de fractal inválido. Escolha 'mandelbrot' ou 'julia'.")
            line_output[x] = fractal(c, z_initial, max_iter)
        return y_idx, line_output

    # Processamento das linhas, com threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(calc_line, y_idx, y) for y_idx, y in enumerate(range(res))]
        for future in futures:
            y_idx, line_output = future.result()
            output[y_idx] = line_output

    return output

if __name__ == '__main__':
    # Servidor (API) roda apenas no Rank 0
    if rank == 0:
        # API executa em uma thread separada, para não sobrecarregar uma única thread
        threading.Thread(target=lambda: app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)).start()

        # Manter a API em execução
        while True:
            time.sleep(1)

    # Processamento é executado no rank 1
    if rank == 1:
        mpi_worker()