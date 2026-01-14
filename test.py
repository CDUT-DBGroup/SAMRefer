#!/usr/bin/env python3
"""
无限循环空转脚本
用于测试或占用CPU资源
"""

import time

def main():
    """无限循环空转"""
    print("开始无限循环空转...")
    print("按 Ctrl+C 可以停止")
    
    counter = 0
    try:
        while True:
            counter += 1
            # 可以添加一些简单的操作，避免完全空转
            if counter % 1000000 == 0:
                print(f"已循环 {counter} 次")
    except KeyboardInterrupt:
        print(f"\n循环已停止，总共执行了 {counter} 次")

if __name__ == '__main__':
    main()

