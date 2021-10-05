# Generated by Django 3.2.6 on 2021-10-05 10:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='sudoku',
            name='target',
            field=models.CharField(blank=True, max_length=81, null=True, verbose_name='target'),
        ),
        migrations.AlterField(
            model_name='sudoku',
            name='photo',
            field=models.ImageField(blank=True, null=True, upload_to='main/picture/', verbose_name='Imagen'),
        ),
    ]
