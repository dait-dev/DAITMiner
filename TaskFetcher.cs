using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace DAITCore
{
    internal class TaskFetcher
    {
        private static readonly HttpClient client = new HttpClient();
        private const string FetchUrl = "https://miningpool.dait.dev/GetMatrix.ashx";
        private const string SubmitUrl = "https://miningpool.dait.dev/SubmitResult.ashx";

        public static async Task<(string, float[,], float[,])> FetchMatricesAsync(string publicKey, int ax, int ay, int bx, int by)
        {
            try
            {
                string json = await client.GetStringAsync(FetchUrl + $"?ax={ax}&ay={ay}&bx={bx}&by={by}" + "&pubKey=" + publicKey);
                var matrices = JsonSerializer.Deserialize<MatrixResponse>(json);
                if (matrices == null) throw new Exception("Failed to deserialize response");
                return (matrices.TaskId, ConvertTo2DArray(matrices.A, ax, ay), ConvertTo2DArray(matrices.B, bx, by));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching matrices: {ex.Message}");
                return (string.Empty, new float[0, 0], new float[0, 0]);
            }
        }

        private static float[,] ConvertTo2DArray(float[] data, int rows, int cols)
        {
            float[,] array = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    array[i, j] = data[i * cols + j];
            return array;
        }
        private static float[] ConvertTo1DArray(float[,] data)
        {
            int rows = data.GetLength(0);
            int cols = data.GetLength(1);
            float[] array = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    array[i * cols + j] = data[i, j];
            return array;
        }
        public static async Task SubmitResultAsync(string taskId, float[,] result, string publicKey)
        {
            try
            {
                var response = new MatrixResult { TaskId = taskId, Result = ConvertTo1DArray(result) };
                string json = JsonSerializer.Serialize(response);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                HttpResponseMessage httpResponse = await client.PostAsync(SubmitUrl + "&pubKey" + publicKey, content);
                Console.WriteLine(await httpResponse.Content.ReadAsStringAsync());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error submitting result: {ex.Message}");
            }
        }
    }
    class MatrixResult
    {
        [JsonPropertyName("taskId")]
        public string TaskId { get; set; } = string.Empty;
        [JsonPropertyName("result")]
        public float[] Result { get; set; } = new float[0];
    }
    class MatrixResponse
    {
        [JsonPropertyName("taskId")]
        public string TaskId { get; set; } = string.Empty;
        [JsonPropertyName("a")]
        public float[] A { get; set; } = new float[0];
        [JsonPropertyName("b")]
        public float[] B { get; set; } = new float[0];
    }
}
