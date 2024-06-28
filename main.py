import asyncio

from aiogram import Bot, Dispatcher, types, F
import logging

from aiogram.filters import CommandStart

from cataract import Cataract

TOKEN = '7369056925:AAFdsIIM4xaNqKhvMqHifOT_uH6GN89z9Z0'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
с = Cataract('cyrillic')
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer('Привет')


@dp.message(F.photo)
async def magic(message: types.Message):
    if message.photo:
        id = f"{message.photo[-1].file_id}.jpg"
        file_name = f"photos/{id}"
        x = await bot.download(message.photo[-1], destination=file_name)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, с.img_to_str, file_name)
        await message.reply(result)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
