# pip install mrjob
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRWeatherAnalysis(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_temps,
                   reducer=self.reducer_get_max_temp),
            MRStep(reducer=self.reducer_find_max_temp_year)
        ]

    def mapper_get_temps(self, _, line):
        # Split the line into components
        parts = line.split(',')
        try:
            year = parts[0]
            temp = float(parts[1])
            yield year, temp
        except ValueError:
            pass  # ignore lines with invalid data

    def reducer_get_max_temp(self, year, temps):
        # Send the max temperature for each year to the same reducer
        yield None, (max(temps), year)

    def reducer_find_max_temp_year(self, _, year_temp_pairs):
        # Find the year with the maximum temperature
        yield max(year_temp_pairs)

if __name__ == '__main__':
    MRWeatherAnalysis.run()


# python Weather.py weather.csv