package mine.design.en.controller;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class MyDemo {

    @Test
    public static void main(String[] args) {
        try {
            System.out.println("start");
            String[] args1=new String[]{"python","C:\\Users\\10541\\Desktop\\LEMarket-master\\enterprise\\oa_web\\src\\main\\webapp\\WEB-INF\\model\\9_30_1.py"};
            Process pr=Runtime.getRuntime().exec(args1);

            BufferedReader in = new BufferedReader(new InputStreamReader(
                    pr.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            pr.waitFor();
            System.out.println("end");
        } catch (Exception e) {
            e.printStackTrace();
        }}
}
